import torch
from torch import nn
import numpy as np
import typing as tp

from .conditioners import MultiConditioner, create_multi_conditioner_from_conditioning_config
from .dit import DiffusionTransformer
from .factory import create_pretransform_from_config
from .pretransforms import Pretransform
from ..inference.generation import generate_diffusion_cond


from time import time

class Profiler:

    def __init__(self):
        self.ticks = [[time(), None]]

    def tick(self, msg):
        self.ticks.append([time(), msg])

    def __repr__(self):
        rep = 80 * "=" + "\n"
        for i in range(1, len(self.ticks)):
            msg = self.ticks[i][1]
            ellapsed = self.ticks[i][0] - self.ticks[i - 1][0]
            rep += msg + f": {ellapsed*1000:.2f}ms\n"
        rep += 80 * "=" + "\n\n\n"
        return rep


class DiffusionModel(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x, t, **kwargs):
        raise NotImplementedError()


class DiffusionModelWrapper(nn.Module):
    def __init__(
                self,
                model: DiffusionModel,
                io_channels,
                sample_size,
                sample_rate,
                min_input_length,
                pretransform: tp.Optional[Pretransform] = None,
    ):
        super().__init__()
        self.io_channels = io_channels
        self.sample_size = sample_size
        self.sample_rate = sample_rate
        self.min_input_length = min_input_length

        self.model = model

        if pretransform is not None:
            self.pretransform = pretransform
        else:
            self.pretransform = None

    def forward(self, x, t, **kwargs):
        return self.model(x, t, **kwargs)


class ConditionedDiffusionModel(nn.Module):
    def __init__(self,
                *args,
                supports_cross_attention: bool = False,
                supports_input_concat: bool = False,
                supports_global_cond: bool = False,
                supports_prepend_cond: bool = False,
                **kwargs):
        super().__init__(*args, **kwargs)
        self.supports_cross_attention = supports_cross_attention
        self.supports_input_concat = supports_input_concat
        self.supports_global_cond = supports_global_cond
        self.supports_prepend_cond = supports_prepend_cond

    def forward(self,
                x: torch.Tensor,
                t: torch.Tensor,
                cross_attn_cond: torch.Tensor = None,
                cross_attn_mask: torch.Tensor = None,
                input_concat_cond: torch.Tensor = None,
                global_embed: torch.Tensor = None,
                prepend_cond: torch.Tensor = None,
                prepend_cond_mask: torch.Tensor = None,
                cfg_scale: float = 1.0,
                cfg_dropout_prob: float = 0.0,
                batch_cfg: bool = False,
                rescale_cfg: bool = False,
                **kwargs):
        raise NotImplementedError()


class ConditionedDiffusionModelWrapper(nn.Module):
    """
    A diffusion model that takes in conditioning
    """
    def __init__(
            self,
            model: ConditionedDiffusionModel,
            conditioner: MultiConditioner,
            io_channels,
            sample_rate,
            min_input_length: int,
            diffusion_objective: tp.Literal["v", "rectified_flow"] = "v",
            pretransform: tp.Optional[Pretransform] = None,
            cross_attn_cond_ids: tp.List[str] = [],
            global_cond_ids: tp.List[str] = [],
            input_concat_ids: tp.List[str] = [],
            prepend_cond_ids: tp.List[str] = [],
            ):
        super().__init__()

        self.model = model
        self.conditioner = conditioner
        self.io_channels = io_channels
        self.sample_rate = sample_rate
        self.diffusion_objective = diffusion_objective
        self.pretransform = pretransform
        self.cross_attn_cond_ids = cross_attn_cond_ids
        self.global_cond_ids = global_cond_ids
        self.input_concat_ids = input_concat_ids
        self.prepend_cond_ids = prepend_cond_ids
        self.min_input_length = min_input_length

    def get_conditioning_inputs(self, conditioning_tensors: tp.Dict[str, tp.Any], negative=False):
        cross_attention_input = None
        cross_attention_masks = None
        global_cond = None
        input_concat_cond = None
        prepend_cond = None
        prepend_cond_mask = None

        # Concatenate all cross-attention inputs over the sequence dimension
        # Assumes that the cross-attention inputs are of shape (batch, seq, channels)
        if len(self.cross_attn_cond_ids) > 0:
            cross_attention_input = []
            cross_attention_masks = []

            for key in self.cross_attn_cond_ids:
                cross_attn_in, cross_attn_mask = conditioning_tensors[key]
                # Add sequence dimension if it's not there
                if len(cross_attn_in.shape) == 2:
                    cross_attn_in = cross_attn_in.unsqueeze(1)
                    cross_attn_mask = cross_attn_mask.unsqueeze(1)
                cross_attention_input.append(cross_attn_in)
                cross_attention_masks.append(cross_attn_mask)

            cross_attention_input = torch.cat(cross_attention_input, dim=1)
            cross_attention_masks = torch.cat(cross_attention_masks, dim=1)


        # Concatenate all global conditioning inputs over the channel dimension
        # Assumes that the global conditioning inputs are of shape (batch, channels)
        if len(self.global_cond_ids) > 0:
            # Concatenate over the channel dimension
            global_cond = torch.cat([conditioning_tensors[key][0] for key in self.global_cond_ids], dim=-1)
            if len(global_cond.shape) == 3:
                global_cond = global_cond.squeeze(1)


        # Concatenate all input concat conditioning inputs over the channel dimension
        # Assumes that the input concat conditioning inputs are of shape (batch, channels, seq)
        if len(self.input_concat_ids) > 0:
            # Concatenate over the channel dimension
            input_concat_cond = torch.cat([conditioning_tensors[key][0] for key in self.input_concat_ids], dim=1)


        # Concatenate all prepend conditioning inputs over the sequence dimension
        # Assumes that the prepend conditioning inputs are of shape (batch, seq, channels)
        if len(self.prepend_cond_ids) > 0:
            prepend_cond = torch.cat([conditioning_tensors[key][0] for key in self.prepend_cond_ids], dim=1)
            prepend_cond_mask = torch.cat([conditioning_tensors[key][1] for key in self.prepend_cond_ids], dim=1)


        if negative:
            return {
                "negative_cross_attn_cond": cross_attention_input,
                "negative_cross_attn_mask": cross_attention_masks,
                "negative_global_cond": global_cond,
                "negative_input_concat_cond": input_concat_cond
            }
        else:

            return {
                "cross_attn_cond": cross_attention_input, # (batch, seq, channels)
                "cross_attn_mask": cross_attention_masks, # (batch, seq)
                "global_cond": global_cond,               # None or (batch, concated_channels)
                "input_concat_cond": input_concat_cond,   # None
                "prepend_cond": prepend_cond,             # None
                "prepend_cond_mask": prepend_cond_mask    # None
            }

    def forward(self, x: torch.Tensor, t: torch.Tensor, cond: tp.Dict[str, tp.Any], **kwargs):
        # print(cond.keys(), cond['cross_attn_cond'].shape, cond['cross_attn_mask'].shape)
        return self.model(x, t, **self.get_conditioning_inputs(cond), **kwargs)


    def generate(self, *args, **kwargs):
        return generate_diffusion_cond(self, *args, **kwargs)


class DiTWrapper(ConditionedDiffusionModel):
    def __init__(
        self,
        *args,
        **kwargs
    ):
        super().__init__(supports_cross_attention=True, supports_global_cond=False, supports_input_concat=False)

        self.model = DiffusionTransformer(*args, **kwargs)

        with torch.no_grad():
            for param in self.model.parameters():
                param *= 0.5

    def forward(self,
                x,
                t,
                cross_attn_cond=None,
                cross_attn_mask=None,
                negative_cross_attn_cond=None,
                negative_cross_attn_mask=None,
                input_concat_cond=None,
                negative_input_concat_cond=None,
                global_cond=None,
                negative_global_cond=None,
                prepend_cond=None,
                prepend_cond_mask=None,
                cfg_scale=1.0,
                cfg_dropout_prob: float = 0.0,
                batch_cfg: bool = True,
                rescale_cfg: bool = False,
                scale_phi: float = 0.0,
                **kwargs):
        # train: cfg_scale = 1, cfg_dropout_prob = 0.1
        # sample: cfg_scale = 7, cfg_dropout_prob = 0

        assert batch_cfg, "batch_cfg must be True for DiTWrapper"
        #assert negative_input_concat_cond is None, "negative_input_concat_cond is not supported for DiTWrapper"
        # print(cross_attn_cond.shape)
        return self.model(
            x,
            t,
            cross_attn_cond=cross_attn_cond,
            cross_attn_cond_mask=cross_attn_mask,
            negative_cross_attn_cond=negative_cross_attn_cond,
            negative_cross_attn_mask=negative_cross_attn_mask,
            input_concat_cond=input_concat_cond,
            prepend_cond=prepend_cond,
            prepend_cond_mask=prepend_cond_mask,
            cfg_scale=cfg_scale,
            cfg_dropout_prob=cfg_dropout_prob,
            scale_phi=scale_phi,
            global_embed=global_cond,
            **kwargs)




def create_diffusion_cond_from_config(config: tp.Dict[str, tp.Any]):


    ############################################## SOME GLOBAL ARGS ##################################################
    model_config = config["model"]
    sample_rate = config.get('sample_rate', None)
    io_channels = model_config.get('io_channels', None)
    
    assert io_channels is not None, "Must specify io_channels in model config"
    assert sample_rate is not None, "Must specify sample_rate in config"
    



    ############################################  DIFFUSION MODEL ####################################################
    diffusion_config = model_config['diffusion']
    diffusion_model_config = diffusion_config.get('config', None)

    assert diffusion_model_config is not None, "Must specify diffusion model config"
    diffusion_model = DiTWrapper(**diffusion_model_config)
    




    ############################################## CONDITIONER ##############################################
    conditioning_config = model_config.get('conditioning', None)

    assert conditioning_config is not None, "Must specify conditioning model config"
    conditioner = create_multi_conditioner_from_conditioning_config(conditioning_config)
    



    ############################################## PRETRANSFORM (VAE) ##############################################
    pretransform_config = model_config.get("pretransform", None)

    assert pretransform_config is not None, "Must specify pretransform model config"
    pretransform = create_pretransform_from_config(pretransform_config, sample_rate)
    min_input_length = pretransform.downsampling_ratio * diffusion_model.model.patch_size
    
    


    
    ############################################# DIFFUSION MODEL ARGS #############################################
    cross_attention_ids = diffusion_config.get('cross_attention_cond_ids', [])
    global_cond_ids = diffusion_config.get('global_cond_ids', [])
    input_concat_ids = diffusion_config.get('input_concat_ids', [])
    prepend_cond_ids = diffusion_config.get('prepend_cond_ids', [])
    diffusion_objective = diffusion_config.get('diffusion_objective', 'v')
    extra_kwargs = {"diffusion_objective":diffusion_objective}
    
    
    


    # Return the proper wrapper class
    return ConditionedDiffusionModelWrapper(
        diffusion_model,
        conditioner,
        min_input_length=min_input_length,
        sample_rate=sample_rate,
        cross_attn_cond_ids=cross_attention_ids,
        global_cond_ids=global_cond_ids,
        input_concat_ids=input_concat_ids,
        prepend_cond_ids=prepend_cond_ids,
        pretransform=pretransform,
        io_channels=io_channels,
        **extra_kwargs
    )