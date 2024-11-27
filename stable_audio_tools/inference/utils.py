from ..data.utils import PadCrop
import subprocess
import torchaudio
import torch
from einops import rearrange
from torchaudio import transforms as T

def set_audio_channels(audio, target_channels):
    if target_channels == 1:
        # Convert to mono
        audio = audio.mean(1, keepdim=True)
    elif target_channels == 2:
        # Convert to stereo
        if audio.shape[1] == 1:
            audio = audio.repeat(1, 2, 1)
        elif audio.shape[1] > 2:
            audio = audio[:, :2, :]
    return audio

def prepare_audio(audio, in_sr, target_sr, target_length, target_channels, device):
    
    audio = audio.to(device)

    if in_sr != target_sr:
        resample_tf = T.Resample(in_sr, target_sr).to(device)
        audio = resample_tf(audio)

    audio = PadCrop(target_length, randomize=False)(audio)

    # Add batch dimension
    if audio.dim() == 1:
        audio = audio.unsqueeze(0).unsqueeze(0)
    elif audio.dim() == 2:
        audio = audio.unsqueeze(0)

    audio = set_audio_channels(audio, target_channels)

    return audio

def save_audio(input_audio, audio_path, sample_rate):
    input_audio = rearrange(input_audio, "b d n -> d (b n)")
    input_audio = input_audio.to(torch.float32).div(torch.max(torch.abs(input_audio))).clamp(-1, 1).mul(32767).to(torch.int16).cpu()
    torchaudio.save(audio_path, input_audio, sample_rate)

def replace_audio(input_video, input_audio, output_video):
    ffmpeg_cmd = [
        'ffmpeg',
        '-i', input_video,        # 输入视频文件
        '-i', input_audio,        # 输入音频文件
        '-c:v', 'copy',           # 复制视频流，不重新编码视频
        '-c:a', 'aac',            # 指定音频编码为aac
        '-map', '0:v:0',          # 从第一个输入（视频）映射视频流
        '-map', '1:a:0',          # 从第二个输入（音频）映射音频流
        '-strict', 'experimental', # 允许使用实验性编码
        '-loglevel', 'error',
        output_video,              # 输出视频文件
        '-y'
    ]

    try:
        subprocess.run(ffmpeg_cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"命令执行失败: {e}")