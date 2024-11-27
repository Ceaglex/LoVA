import torch
from torch import nn


class Bottleneck(nn.Module):
    def __init__(self, is_discrete: bool = False):
        super().__init__()

        self.is_discrete = is_discrete

    def encode(self, x, return_info=False, **kwargs):
        raise NotImplementedError

    def decode(self, x):
        raise NotImplementedError


def vae_sample(mean, scale):
        stdev = nn.functional.softplus(scale) + 1e-4
        var = stdev * stdev
        logvar = torch.log(var)
        latents = torch.randn_like(mean) * stdev + mean

        kl = (mean * mean + var - logvar - 1).sum(1).mean()

        return latents, kl

class VAEBottleneck(Bottleneck):
    def __init__(self):
        super().__init__(is_discrete=False)

    def encode(self, x, return_info=False, **kwargs):
        info = {}

        mean, scale = x.chunk(2, dim=1)

        x, kl = vae_sample(mean, scale)

        info["kl"] = kl

        if return_info:
            return x, info
        else:
            return x

    def decode(self, x):
        return x

def compute_mean_kernel(x, y):
        kernel_input = (x[:, None] - y[None]).pow(2).mean(2) / x.shape[-1]
        return torch.exp(-kernel_input).mean()

def compute_mmd(latents):
    latents_reshaped = latents.permute(0, 2, 1).reshape(-1, latents.shape[1])
    noise = torch.randn_like(latents_reshaped)

    latents_kernel = compute_mean_kernel(latents_reshaped, latents_reshaped)
    noise_kernel = compute_mean_kernel(noise, noise)
    latents_noise_kernel = compute_mean_kernel(latents_reshaped, noise)
    
    mmd = latents_kernel + noise_kernel - 2 * latents_noise_kernel
    return mmd.mean()




    