from clip import CLIP
from diffusion import Diffusion
from encoder import VAE_Encoder
from decoder import VAE_Decoder

import model_converter

def preload_models_from_standard_weights(ckpt_path, device):
    state_dict = model_converter.load_from_standard_weights(ckpt_path, device)

    encoder = VAE_Encoder().to(device)
    encoder.load_state_dict(state_dict['encoder'], strict=True)  # load_state_dict is a method of nn.Module

    decoder = VAE_Decoder().to(device)
    decoder.load_state_dict(state_dict['decoder'], strict=True)  # strict means the names should match exactly

    diffusion = Diffusion().to(device)
    diffusion.load_state_dict(state_dict['diffusion'], strict=True)

    clip = CLIP().to(device)
    clip.load_state_dict(state_dict['clip'], strict=True)

    return {
        "encoder": encoder,
        "decoder": decoder,
        "diffusion": diffusion,
        "clip": clip
    }