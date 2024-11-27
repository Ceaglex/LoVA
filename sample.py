import json
import pytorch_lightning as pl
from einops import rearrange
import torch
import torchaudio
from torch.utils.data import DataLoader
import safetensors
from safetensors.torch import load_file
import random
from moviepy.editor import VideoFileClip

import shutil
from collections import OrderedDict
from datetime import datetime
import os
import torch.nn.functional as F

from stable_audio_tools import create_model_from_config, replace_audio, save_audio
from stable_audio_tools.data.dataset import VideoFeatDataset, VideoFeatDataset_VL, collation_fn
from stable_audio_tools.training.training_wrapper import DiffusionCondTrainingWrapper
from stable_audio_tools.inference.generation import generate_diffusion_cond, generate_diffusion_cond_from_path


device = 0
model_config_file = './stable_audio_tools/configs/model_config_vl30.json'
model_weight = './weight/epoch=60-step=2818.safetensors'

with open(model_config_file) as f:
    model_config = json.load(f)
    sample_rate = model_config["sample_rate"]
    sample_size = model_config["sample_size"]
    fps = model_config["fps"]
    variable_length = model_config["variable_length"]
    force_channels = "stereo" if model_config["audio_channels"] == 2 else "mono"

state_dict = load_file(model_weight)
model = create_model_from_config(model_config)
model.load_state_dict(state_dict, strict=True)
print(sample_size, sample_rate, sample_size/sample_rate, variable_length, fps, force_channels)



conditioning = {
    'feature': ['./asset/demo_video.mp4']
}
video_clip = VideoFileClip(conditioning['feature'][0])
seconds_total = int(video_clip.duration)
print(f"Video total duration: {seconds_total}")
# seconds_total = 10

output = generate_diffusion_cond(
    model = model.to(device),
    steps=150,
    cfg_scale=7,
    conditioning=conditioning,
    sample_size=int(sample_rate*seconds_total),
    batch_size=len(conditioning['feature']),
    sigma_min=0.3,
    sigma_max=500,
    sampler_type="dpmpp-3m-sde",
    device=device
)

audio_path = "./asset/demo_audio.wav"
waveform = output[0:1,...,:int(seconds_total*sample_rate)]
save_audio(waveform, audio_path, sample_rate)
print(f"Audio Successfully saved in {audio_path}")