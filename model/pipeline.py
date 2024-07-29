import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from ddpm import DDPMSampler

WIDTH = 512
HEIGHT = 512
LATENTS_WIDTH = 512 // 8
LATENTS_HEIGHT = 512 // 8

def generate(prompt, uncond_prompt = None, input_image = None, 
             strength = 0.8, do_cfg = True, cfg_scale = 7.5, 
             sampler_name = 'ddpm', 
             n_inference_steps = 50, models = {}, seed = None,
             device = None, idle_device = None, tokenizer = None):
    with torch.no_grad():

        if not 0 < strength <= 1:
            raise ValueError('Strength must be between 0 and 1')
        # if we want to move things to the cpu

        if idle_device:
            to_idle: lambda x: x.to(idle_device)
        else:   
            to_idle = lambda x: x
        generator = torch.Generator(device=device)

        if seed is None:
            generator.seed()
        else:
            generator.manual_seed(seed)
        clip = models['clip']
        clip.to(device)
        
        if do_cfg:
            # Convert prompt into tokens
            cond_tokens = tokenizer.batch_encode_plus([prompt], padding = "max_length", max_length = 77).input_ids  # padding means that the prompt will be padded to the max length of the batch
            # (batch_size, seq_length)
            cond_tokens = torch.tensor(cond_tokens, dtype = torch.long, device = device)
            # (batch_size, seq_length, d_embd)
            cond_context = clip(cond_tokens)

            uncond_tokens = tokenizer.batch_encode_plus([uncond_prompt], padding = "max_length", max_length = 77).input_ids
            uncond_tokens = torch.tensor(uncond_tokens, dtype = torch.long, device = device)
            uncond_context = clip(uncond_tokens)
            # (2 * batch_size, seq_length, d_embd)
            context = torch.cat([cond_context, uncond_context])
        else:
            tokens = tokenizer.batch_encode_plus([prompt], padding = "max_length", max_length = 77).input_ids
            tokens = torch.tensor(tokens, dtype = torch.long, device = device)
            # (batch_size, seq_length, d_embd)
            context = clip(tokens)
        to_idle(clip) # move clip back to the cpu

        if sampler_name == "ddpm":
            sampler = DDPMSampler(generator)
            sampler.set_inference_steps(n_inference_steps)
        else:
            raise ValueError(f"Sampler {sampler_name} not supported")
        
        latents_shape = (1, 4, LATENTS_HEIGHT, LATENTS_WIDTH)

        if input_image:
            encoder = models['encoder']
            encoder.to(device)

            input_image_tensor = input_image.resize((WIDTH, HEIGHT))
            input_image_tensor = np.array(input_image_tensor)
            # (Height, Width, Channel) RGB gives 3 channels
            input_image_tensor = torch.tensor(input_image_tensor, dtype = torch.float32, device = device)
            input_image_tensor = rescale(input_image_tensor, (0, 255), (-1, 1))
            # (1, Height, Width, Channel)
            input_image_tensor = input_image_tensor.unsqueeze(0)
            # (1, Channel, Height, Width)
            input_image_tensor = input_image_tensor.permute(0, 3, 1, 2)

            # Put the image into the VAE encoder
            encoder_noise = torch.randn(latents_shape, generator=generator, device = device)
            latent = encoder(input_image_tensor, encoder_noise)

            sampler.set_strength(strength = strength)
            latent = sampler.add_noise(latent, sampler.timesteps[0])

            to_idle(encoder)
        else:
            latent = torch.randn(latents_shape, generator=generator, device=device)

        diffusion = models['diffusion']
        diffusion.to(device)

        timesteps = tqdm(sampler.timesteps)  # tqdm is a progress bar
        for i, timestep in enumerate(timesteps):
            # to (1, 320)
            time_embedding = get_time_embedding(timestep).to(device)

            # (batch_size, 4, latents_height, latents_width)
            model_input = latent

            if do_cfg:
                # (batch_size * 2, 4, latents_height, latents_width)
                model_input = model_input.repeat(2, 1, 1, 1)  # one with the prompt one without the prompt

            # output is the predicted noise by the UNET
            model_output = diffusion(model_input, context, time_embedding)

            if do_cfg:
                output_cond, output_uncond = model_output.chunk(2)
                model_output = cfg_scale * (output_cond - output_uncond) + output_uncond

            latent = sampler.step(timestep, latent, model_output)
        to_idle(diffusion)

        decoder = models['decoder']
        decoder.to(device)
        images = decoder(latent)
        to_idle(decoder)

        images = rescale(images, (-1, 1), (0, 255), clamp = True)
        # (batch_size, channel, height, width) -> (batch_size, height, width, channel)
        images = images.permute(0, 2, 3, 1)
        images = images.to("cpu", torch.uint8).numpy()
        return images[0]  # [] is the batch size 
    
def rescale(x, old_range, new_range, clamp = False):
    old_min, old_max = old_range
    new_min, new_max = new_range
    x -= old_min
    x = x * (new_max - new_min) / (old_max - old_min) + new_min
    if clamp:
        x = x.clamp(new_min, new_max)
    return x

# Convert the timestep into a time embedding
def get_time_embedding(timestep: int):
    freqs = torch.pow(10000, -torch.arange(0, 160, dtype=torch.float32) / 160)  # d_time = 320, 160 = d_time / 2
    x = torch.tensor([timestep], dtype=torch.float32)[:, None] * freqs[None]  # [:, None] add a dimension, freqs[None] add a dimension
    return torch.cat([torch.cos(x), torch.sin(x)], dim=-1)
