# from clip import CLIP
# from diffusion import Diffusion
# from encoder import VAE_Encoder
# from decoder import VAE_Decoder

# import model_converter

# def preload_models_from_standard_weights(ckpt_path, device):
#     state_dict = model_converter.load_from_standard_weights(ckpt_path, device)

#     encoder = VAE_Encoder().to(device)
#     encoder.load_state_dict(state_dict["encoder"], strict=True)  # load_state_dict is a method of nn.Module

#     decoder = VAE_Decoder().to(device)
#     decoder.load_state_dict(state_dict["decoder"], strict=True)  # strict means the names should match exactly

#     diffusion = Diffusion().to(device)
#     diffusion.load_state_dict(state_dict['diffusion'], strict=True)

#     clip = CLIP().to(device)
#     clip.load_state_dict(state_dict['clip'], strict=True)

#     return {
#         "encoder": encoder,
#         "decoder": decoder,
#         "diffusion": diffusion,
#         "clip": clip
#     }


from clip import CLIP
from diffusion import Diffusion
from encoder import VAE_Encoder
from decoder import VAE_Decoder

import model_converter

def preload_models_from_standard_weights(ckpt_path, device):
    state_dict = model_converter.load_from_standard_weights(ckpt_path, device)

    encoder = VAE_Encoder().to(device)
    missing_keys, unexpected_keys = encoder.load_state_dict(state_dict["encoder"], strict=True)
    if missing_keys or unexpected_keys:
        print(f"Encoder missing keys: {missing_keys}")
        print(f"Encoder unexpected keys: {unexpected_keys}")

    decoder = VAE_Decoder().to(device)
    missing_keys, unexpected_keys = decoder.load_state_dict(state_dict["decoder"], strict=True)
    if missing_keys or unexpected_keys:
        print(f"Decoder missing keys: {missing_keys}")
        print(f"Decoder unexpected keys: {unexpected_keys}")

    diffusion = Diffusion().to(device)
    missing_keys, unexpected_keys = diffusion.load_state_dict(state_dict['diffusion'], strict=True)
    if missing_keys or unexpected_keys:
        print(f"Diffusion missing keys: {missing_keys}")
        print(f"Diffusion unexpected keys: {unexpected_keys}")

    clip = CLIP().to(device)
    missing_keys, unexpected_keys = clip.load_state_dict(state_dict['clip'], strict=True)
    if missing_keys or unexpected_keys:
        print(f"CLIP missing keys: {missing_keys}")
        print(f"CLIP unexpected keys: {unexpected_keys}")

    return {
        "encoder": encoder,
        "decoder": decoder,
        "diffusion": diffusion,
        "clip": clip
    }
