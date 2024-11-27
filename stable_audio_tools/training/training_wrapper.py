import pytorch_lightning as pl
import os
import torch
import typing as tp
import wandb
import shutil
from pytorch_lightning.utilities.rank_zero import rank_zero_only
from safetensors.torch import save_file
from torch.nn import functional as F

from .. import  replace_audio, save_audio
from ..inference.sampling import get_alphas_sigmas, sample, sample_discrete_euler
from ..inference.generation import generate_diffusion_cond
from ..models.diffusion import ConditionedDiffusionModelWrapper
from .losses import MSELoss, MSELoss_wDuration, MultiLoss
from .utils import create_optimizer_from_config, create_scheduler_from_config







class DiffusionCondTrainingWrapper(pl.LightningModule):
    '''
    Wrapper for training a conditional audio diffusion model.
    '''
    def __init__(
            self,
            model: ConditionedDiffusionModelWrapper,
            lr: float = None,
            optimizer_configs: dict = None,
            pre_encoded: bool = False,
            cfg_dropout_prob = 0.1,
            duration_mask = "seconds_total",
            latent_per_sec = 21.5,
            timestep_sampler: tp.Literal["uniform", "logit_normal"] = "uniform",
    ):
        super().__init__()

        self.diffusion = model

        self.cfg_dropout_prob = cfg_dropout_prob

        self.rng = torch.quasirandom.SobolEngine(1, scramble=True)

        self.timestep_sampler = timestep_sampler

        self.diffusion_objective = model.diffusion_objective


        self.loss_modules = [
            MSELoss_wDuration(
                "output", 
                "targets", 
                duration_mask,    # duration_mask不设置成none时，只有部分latent会被用于loss计算
                weight=1.0, 
                mask_key=None, 
                name="mse_loss",
                latent_per_sec = latent_per_sec # sample rate/20
            )
        ]

        # self.loss_modules = [
        #     MSELoss("output", 
        #            "targets", 
        #            weight=1.0, 
        #            mask_key=None, 
        #            name="mse_loss"
        #     )
        # ]

        self.losses = MultiLoss(self.loss_modules)

        self.train_epoch_losses_info = []
        
        self.test_epoch_losses_info = []

        assert lr is not None or optimizer_configs is not None, "Must specify either lr or optimizer_configs in training config"

        if optimizer_configs is None:
            optimizer_configs = {
                "diffusion": {
                    "optimizer": {
                        "type": "Adam",
                        "config": {
                            "lr": lr
                        }
                    }
                }
            }
        else:
            if lr is not None:
                print(f"WARNING: learning_rate and optimizer_configs both specified in config. Ignoring learning_rate and using optimizer_configs.")

        self.optimizer_configs = optimizer_configs
        self.pre_encoded = pre_encoded


    def configure_optimizers(self):
        diffusion_opt_config = self.optimizer_configs['diffusion']


        trainable_params = self.diffusion.parameters()
        
        trainable_params = []
        for name, param in self.diffusion.named_parameters():
            if ('pos_emb' in name ) and param.requires_grad:  #
                trainable_params.append(param)
                print(name)
            if ('layers' in name ) and param.requires_grad:  #
                idx = name[name.find('layers'):]
                idx = int(idx.split('.')[1])
                if idx >= 23:
                    trainable_params.append(param)
                    print(name)

        
        opt_diff = create_optimizer_from_config(diffusion_opt_config['optimizer'], trainable_params)

        if "scheduler" in diffusion_opt_config:
            sched_diff = create_scheduler_from_config(diffusion_opt_config['scheduler'], opt_diff)
            sched_diff_config = {
                "scheduler": sched_diff,
                "interval": "step"
            }
            return [opt_diff], [sched_diff_config]

        return [opt_diff]


    def training_step(self, batch, batch_idx):
        reals, metadata = batch
        # print(reals.shape, metadata.keys(), metadata['seconds_total'], metadata['seconds_total'].shape)
        loss_info = {}

        diffusion_input = reals  # [batchsize, hidden_dim(VAE), seq]

        with torch.cuda.amp.autocast():
            conditioning = self.diffusion.conditioner(metadata, self.device) # dict{"condition_name":(condition,  mask)}   ([batchsize, seq, output_dim], [batchsize, seq]) 


        if self.diffusion.pretransform is not None:
            self.diffusion.pretransform.to(self.device)

            if not self.pre_encoded:
                with torch.cuda.amp.autocast() and torch.set_grad_enabled(self.diffusion.pretransform.enable_grad):
                    diffusion_input = self.diffusion.pretransform.encode(diffusion_input)
            else:            
                # Apply scale to pre-encoded latents if needed, as the pretransform encode function will not be run
                if hasattr(self.diffusion.pretransform, "scale") and self.diffusion.pretransform.scale != 1.0:
                    diffusion_input = diffusion_input / self.diffusion.pretransform.scale

        if self.timestep_sampler == "uniform":   
            # Draw uniformly distributed continuous timesteps
            t = self.rng.draw(reals.shape[0])[:, 0].to(self.device) # [batchsize]
        elif self.timestep_sampler == "logit_normal":
            t = torch.sigmoid(torch.randn(reals.shape[0], device=self.device))
            

        # Calculate the noise schedule parameters for those timesteps
        if self.diffusion_objective == "v":
            alphas, sigmas = get_alphas_sigmas(t)
        elif self.diffusion_objective == "rectified_flow":
            alphas, sigmas = 1-t, t


        # Combine the ground truth data and the noise
        alphas = alphas[:, None, None]
        sigmas = sigmas[:, None, None]
        noise = torch.randn_like(diffusion_input)
        noised_inputs = diffusion_input * alphas + noise * sigmas


        if self.diffusion_objective == "v":
            targets = noise * alphas - diffusion_input * sigmas
        elif self.diffusion_objective == "rectified_flow":
            targets = noise - diffusion_input


        extra_args = {}
        with torch.cuda.amp.autocast():
            output = self.diffusion(noised_inputs, t, cond=conditioning, cfg_dropout_prob = self.cfg_dropout_prob, **extra_args)
            loss_info.update({
                "output": output,
                "targets": targets,
                "seconds_total":metadata['seconds_total'],
                "padding_mask":  None,
            })
            loss, losses = self.losses(loss_info)


        log_dict = {
            'step': batch_idx,
            'train_step/loss': loss.detach(),
            'train_step/lr': self.trainer.optimizers[0].param_groups[0]['lr']
            # 'train/std_data': diffusion_input.std(),
        }
        
        for loss_name, loss_value in losses.items():
            log_dict[f"train/{loss_name}"] = loss_value.detach()
        self.train_epoch_losses_info.append(log_dict)
        self.log_dict(log_dict)
        
        return loss

    def on_train_epoch_end(self):
        train_epoch_losses_info = self.all_gather(self.train_epoch_losses_info)
        train_epoch_loss = torch.stack([loss['train_step/loss'] for loss in train_epoch_losses_info])
        train_epoch_lr = torch.stack([loss['train_step/lr'] for loss in train_epoch_losses_info])

        log_dict = {
            'epoch': self.current_epoch,
            'train_epoch/loss':train_epoch_loss.mean(),
            'train_epoch/lr':train_epoch_lr.mean()
            }
        self.log_dict(log_dict)
        self.epoch_losses_info = []
        print(f"Epoch:{self.current_epoch}   Loss:{train_epoch_loss.mean()}")

    def validation_step(self, batch, batch_idx):
        pass
    def on_validation_epoch_end(self):
        pass


    def export_model(self, path, use_safetensors=True):
        if use_safetensors:
            save_file(self.diffusion.state_dict(), path)
        else:
            torch.save({"state_dict": self.diffusion.state_dict()}, path)




class DiffusionCondDemoCallback(pl.Callback):
    '''
    Upload media table to wandb
    '''
    def __init__(self, 
                 test_dataloader,
                 sample_size = None,
                 every_n_epochs = 1,
                 sample_rate = 44100,
                 output_dir = "./demo/temp"):
        super().__init__()
        self.sample_size = sample_size
        self.every_n_epochs = every_n_epochs
        self.test_dataloader = test_dataloader
        self.sample_rate = sample_rate
        self.output_dir = output_dir


    @rank_zero_only
    @torch.no_grad()
    def on_train_epoch_end(self, trainer, module: DiffusionCondTrainingWrapper) -> None:  
        if module.current_epoch % self.every_n_epochs != 0:
            return
        try:
            module.eval()
            # table = wandb.Table(columns=['gt_video', "gen_video", "gen_audio"])
            table = wandb.Table(columns=["gen_video", "gen_audio", "name"])
                
            for audio, conditioning in self.test_dataloader:
                # if self.sample_size == None:
                #     sample_size = int(self.sample_rate*max(conditioning['seconds_total']))
                # else:
                #     sample_size = self.sample_size
                # print(sample_size, end = " ")
                sample_size = int(self.sample_rate*max(conditioning['seconds_total'])) if self.sample_size == None else self.sample_size
                print("sample_size: ", sample_size)
                output = generate_diffusion_cond(
                        model = module.diffusion,
                        steps=150,
                        cfg_scale=7,
                        conditioning=conditioning,
                        sample_size= sample_size,
                        batch_size=len(conditioning['feature']),
                        sigma_min=0.3,
                        sigma_max=500,
                        sampler_type="dpmpp-3m-sde", # k-dpm-fast
                        device=module.device
                )

                for idx in range(len(audio)):
                    if 'AuidoSet' in conditioning['video_path'][idx] or 'AudioSet' in conditioning['video_path'][idx]:
                        l = conditioning['video_path'][idx].split('/')
                        video_path = os.path.join('/home/chengxin/chengxin/AudioSet/dataset/', l[-4], l[-2], l[-1])
                    else:
                        video_path = conditioning['video_path'][idx].replace('../../../', '/home/chengxin/chengxin/')
                        # video_path = conditioning['video_path'][idx].replace('../../', './')
                    gt_video = wandb.Video(video_path, fps=4)

                    audio_path = f"{self.output_dir}/{video_path.split('/')[-1].replace('.mp4', '.wav')}"
                    waveform = output[idx:1+idx,...,:int(conditioning['seconds_total'][idx]*self.sample_rate)]

                    save_audio(waveform, audio_path, self.sample_rate)
                    gen_audio = wandb.Audio(audio_path, sample_rate=self.sample_rate)

                    moved_video_path = f"{self.output_dir}/{video_path.split('/')[-1]}"
                    shutil.copy(video_path, moved_video_path)
                    generated_video_path = moved_video_path.replace(".mp4","_GEN.mp4")
                    replace_audio(moved_video_path, audio_path, generated_video_path)
                    gen_video = wandb.Video(generated_video_path, fps=4)

                    # table.add_data(gt_video, gen_video, gen_audio)
                    table.add_data(gen_video, gen_audio, video_path.split('/')[-1])

            trainer.logger.experiment.log({f"table{module.current_epoch}":table})
            torch.cuda.empty_cache()
        except Exception as e:
            print(e)
            pass

        
