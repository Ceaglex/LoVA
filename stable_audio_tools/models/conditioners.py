#Heavily influenced by https://github.com/facebookresearch/audiocraft/blob/main/audiocraft/modules/conditioners.py

import torch
import typing as tp

from .adp import NumberEmbedder
import torch
import logging, warnings

from torch import nn
import clip
from moviepy.editor import VideoFileClip
from PIL import Image
import torch.nn.functional as F


class Conditioner(nn.Module):
    def __init__(
            self,
            dim: int,
            output_dim: int,
            project_out: bool = False
            ):
        
        super().__init__()

        self.dim = dim
        self.output_dim = output_dim
        self.proj_out = nn.Linear(dim, output_dim) if (dim != output_dim or project_out) else nn.Identity()

    def forward(self, x: tp.Any) -> tp.Any:
        raise NotImplementedError()
    
    
class T5Conditioner(Conditioner):

    T5_MODELS = ["t5-small", "t5-base", "t5-large", "t5-3b", "t5-11b",
              "google/flan-t5-small", "google/flan-t5-base", "google/flan-t5-large",
              "google/flan-t5-xl", "google/flan-t5-xxl"]
    
    T5_MODEL_DIMS = {
        "t5-small": 512,
        "t5-base": 768,
        "t5-large": 1024,
        "t5-3b": 1024,
        "t5-11b": 1024,
        "t5-xl": 2048,
        "t5-xxl": 4096,
        "google/flan-t5-small": 512,
        "google/flan-t5-base": 768,
        "google/flan-t5-large": 1024,
        "google/flan-t5-3b": 1024,
        "google/flan-t5-11b": 1024,
        "google/flan-t5-xl": 2048,
        "google/flan-t5-xxl": 4096,
    }

    def __init__(
            self,
            output_dim: int,
            t5_model_name: str = "t5-base",
            max_length: str = 128,
            enable_grad: bool = False,
            project_out: bool = False
    ):
        assert t5_model_name in self.T5_MODELS, f"Unknown T5 model name: {t5_model_name}"
        super().__init__(self.T5_MODEL_DIMS[t5_model_name], output_dim, project_out=project_out)
        
        from transformers import T5EncoderModel, AutoTokenizer

        self.max_length = max_length
        self.enable_grad = enable_grad

        # Suppress logging from transformers
        previous_level = logging.root.manager.disable
        logging.disable(logging.ERROR)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                # self.tokenizer = T5Tokenizer.from_pretrained(t5_model_name, model_max_length = max_length)
                # model = T5EncoderModel.from_pretrained(t5_model_name, max_length=max_length).train(enable_grad).requires_grad_(enable_grad)
                self.tokenizer = AutoTokenizer.from_pretrained(t5_model_name)
                model = T5EncoderModel.from_pretrained(t5_model_name).train(enable_grad).requires_grad_(enable_grad).to(torch.float16)
            finally:
                logging.disable(previous_level)
            
        if self.enable_grad:
            self.model = model
        else: 
            self.__dict__["model"] = model


    def forward(self, texts: tp.List[str], device: tp.Union[torch.device, str]) -> tp.Tuple[torch.Tensor, torch.Tensor]:
        
        self.model.to(device)
        self.proj_out.to(device)

        encoded = self.tokenizer(
            texts,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )

        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device).to(torch.bool)
        # print(input_ids)
        # print(input_ids.shape)

        self.model.eval()
        
        with torch.cuda.amp.autocast(dtype=torch.float16) and torch.set_grad_enabled(self.enable_grad):
            embeddings = self.model(
                input_ids=input_ids, attention_mask=attention_mask
            )["last_hidden_state"]    
        
        embeddings = self.proj_out(embeddings.float())
        # print(embeddings.shape)


        embeddings = embeddings * attention_mask.unsqueeze(-1).float()
        # embeddings = embeddings[:,:31,:]
        # print(embeddings.shape)    

        return embeddings, attention_mask

class NumberConditioner(Conditioner):
    '''
        Conditioner that takes a list of floats, normalizes them for a given range, and returns a list of embeddings
    '''
    def __init__(self, 
                output_dim: int,
                min_val: float=0,
                max_val: float=1
                ):
        super().__init__(output_dim, output_dim)

        self.min_val = min_val
        self.max_val = max_val

        self.embedder = NumberEmbedder(features=output_dim)

    def forward(self, floats: tp.List[float], device=None) -> tp.Any:
    
            # Cast the inputs to floats
            floats = [float(x) for x in floats]

            floats = torch.tensor(floats).to(device)

            floats = floats.clamp(self.min_val, self.max_val)
    
            normalized_floats = (floats - self.min_val) / (self.max_val - self.min_val)

            # Cast floats to same type as embedder
            embedder_dtype = next(self.embedder.parameters()).dtype
            normalized_floats = normalized_floats.to(embedder_dtype)

            float_embeds = self.embedder(normalized_floats).unsqueeze(1)

            # print(floats, float_embeds.shape)
            return [float_embeds, torch.ones(float_embeds.shape[0], 1).to(device)]


class TimeAlignConditioner(Conditioner):
    def __init__(self, 
                output_dim: int,):
        super().__init__(output_dim, output_dim)
        self.output_dim = output_dim
    def forward(self, Time_info: tp.List[torch.Tensor], device=None) -> tp.Any:
        padded_length = self.output_dim
        if padded_length == None:
            padded_length = max([len(t) for t in Time_info])
        for i in range(len(Time_info)):
            time_cond = Time_info[i].to(device)
            length = len(time_cond)
            time_cond = F.pad(time_cond, (0,padded_length-length), "constant", -1)
            # time_cond = torch.tensor([time_cond[int(i / padded_length * length)] for i in range(padded_length)]).to(device)
            Time_info[i] = time_cond
        Time_info = torch.stack(Time_info).unsqueeze(1).to(device) # [batchsize, 1, padded_length]
        # print(padded_length, Time_info.shape)
        return [Time_info, torch.ones(Time_info.shape[0], 1).to(device)]
        
        

class CLIPFeatConditioner(Conditioner):
    def __init__(self,
                 output_dim :int,
                 fps :int = 22,
                 clip_name :str = None,
                 download_root :str = None,
                 project_out: bool = False,
                 **kargs):
        '''
            Get the video feature from restored feature or videos
        '''
        super().__init__(768, output_dim, project_out=project_out)
        self.fps = fps
        self.clip_name = clip_name
        self.download_root = download_root
        # self.clip_model = None
        # self.preprocess = None

    def forward(self, data, device=None) -> tp.Any:
        if isinstance(data[0], str):
            with torch.no_grad():
                # if self.clip_model == None:
                clip_model,  preprocess = clip.load(self.clip_name, device=device, download_root=self.download_root)
                # elif next(self.clip_model.parameters()).device.index != device:
                clip_model.to(device)

                images_features = []
                for mp4_file in data:
                    video = VideoFileClip(mp4_file)
                    video = video.set_fps(self.fps)
                    print(f"Extracting features from video:{mp4_file}")

                    frames = []
                    image_features = []
                    for idx, frame in enumerate(video.iter_frames()):
                        frame = Image.fromarray(frame)
                        frame = preprocess(frame).unsqueeze(0)
                        frames.append(frame)

                        # do it in batch to avoid OOM
                        if idx % 240 == 239  or  idx == int(video.duration*video.fps)-1:
                            frames = torch.cat(frames).to(device)
                            frame_batchsize = 20
                            for i in range(len(frames)//frame_batchsize + 1):
                                start_id = i*frame_batchsize
                                end_id = (i+1)*frame_batchsize
                                if start_id >= len(frames):
                                    break
                                image_features.append(clip_model.encode_image(frames[start_id:end_id]))
                            # del frames
                            # torch.cuda.empty_cache()
                            frames = []

                    image_features = torch.concat(image_features).to(device)
                    # text_tokens = clip.tokenize(['Dark and silence' for _ in range(image_features.shape[0])]).to(device)
                    # text_features  = clip_model.encode_text(text_tokens)
                    # image_features = text_features
                    images_features.append(image_features)                    

                max_length = max([image_features.shape[0] for image_features in images_features])
                images_features = torch.stack([
                    F.pad(image_features, (0, 0, 0, max_length - image_features.shape[0]), mode='constant', value=0) 
                    for image_features in images_features
                ])

                return [images_features.to(device),               torch.ones(images_features.shape[0], images_features.shape[1]).to(device)]
        
        elif isinstance(data[0], torch.Tensor):
            # TODO: attention MASK ?????
            # [batchsize, frame_num, output_dim]   [batchsize, frame_num]
            return [data.to(device),               torch.ones(data.shape[0], data.shape[1]).to(device)]  
        
        else:
            raise ValueError("data type for CLIPConditioner can only be str(directory path) or tensor(restored feature)")


class MultiConditioner(nn.Module):
    """
    A module that applies multiple conditioners to an input dictionary based on the keys

    Args:
        conditioners: a dictionary of conditioners with keys corresponding to the keys of the conditioning input dictionary (e.g. "prompt")
        default_keys: a dictionary of default keys to use if the key is emmpty 
                      (e.g. {"feature": "video_path"}, When the dict['feature'] is empty, use dict['video_path'] as the input to the conditioner)
    """
    def __init__(self, 
                 conditioners: tp.Dict[str, Conditioner], 
                 ):
        super().__init__()

        self.conditioners = nn.ModuleDict(conditioners)

    def forward(self, batch_metadata: tp.Dict[str, tp.Any], device: tp.Union[torch.device, str]) -> tp.Dict[str, tp.Any]:
        output = {}
        for key, conditioner in self.conditioners.items():
            if key in batch_metadata: #  and len(batch_metadata[key]) != 0
                output[key] = conditioner(batch_metadata[key], device)
            else:
                continue
            
        
        if len(output) == 0:
            print("No conditioned is specified !!")
            raise    
        return output
    
    
def create_multi_conditioner_from_conditioning_config(config: tp.Dict[str, tp.Any]) -> MultiConditioner:
    """
    Create a MultiConditioner from a conditioning config dictionary

    Args:
        config: the conditioning config dictionary
        device: the device to put the conditioners on
    """
    conditioners = {}
    cond_dim = config["cond_dim"]

    for conditioner_info in config["configs"]:
        id = conditioner_info["id"]
        conditioner_type = conditioner_info["type"]

        conditioner_config = {"output_dim": cond_dim}
        conditioner_config.update(conditioner_info["config"])

        if conditioner_type == "number":
            conditioners[id] = NumberConditioner(**conditioner_config)
        elif conditioner_type == "video_feature":
            conditioners[id] = CLIPFeatConditioner(**conditioner_config)
        elif conditioner_type == "t5":
            conditioners[id] = T5Conditioner(**conditioner_config)
        elif conditioner_type == "time_align":
            conditioners[id] = TimeAlignConditioner(**conditioner_config)

    return MultiConditioner(conditioners)