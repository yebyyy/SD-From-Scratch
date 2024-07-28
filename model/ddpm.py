import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class DDPMSampler:

    def __init__(self, generator: torch.Generator, num_training_steps = 1000, Beta_start: float = 0.00085, Beta_end: float = 0.0120):
        self.betas = torch.linspace(Beta_start ** 0.5, Beta_end ** 0.5, num_training_steps, dtype=torch.float32) ** 2  # ^2 to get the actual beta value
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim = 0)  # Cumulative product [alpha_0, alpha_0 * alpha_1, alpha_0 * alpha_1 * alpha_2, ...]
        self.one = torch.tensor(1.0)
        self.generator = generator
        self.num_training_steps = num_training_steps
        self.timesteps = torch.from_numpy(np.arange(0, num_training_steps)[::-1].copy())  # [num_training_steps - 1, num_training_steps - 2, ..., 0], copy is used to avoid memory overlap

    def set_inference_steps(self, num_inference_steps=50):
        self.num_inference_steps = num_inference_steps
        ### 999, 999 - 20, 999 - 40, ..., 0  a total of 50 steps
        step_ratio = self.num_training_steps // num_inference_steps
        self.timesteps = torch.from_numpy(np.arange(0, num_inference_steps) * step_ratio.round()[::-1].copy().astype(np.int64))

    def add_noise(self, original_samples: torch.FloatTensor, timestep: torch.IntTensor) -> torch.FloatTensor:
        alpha_cumprod = self.alphas_comprod.to(original_samples.device, dtype = original_samples.dtype)
        timestep = timestep.to(original_samples.device)
        sqrt_alpha_cumprod = alpha_cumprod[timestep] ** 0.5
        sqrt_alpha_cumprod = sqrt_alpha_cumprod.flatten()  # flatten is used to remove the extra dimension
        while len(sqrt_alpha_cumprod.shape) < len(original_samples.shape):
            sqrt_alpha_cumprod = sqrt_alpha_cumprod.unsqueeze(-1)
        
        sqrt_one_minus_alpha_cumprod = (1 - alpha_cumprod[timestep]) ** 0.5
        sqrt_one_minus_alpha_cumprod = sqrt_one_minus_alpha_cumprod.flatten()
        while len(sqrt_one_minus_alpha_cumprod.shape) < len(original_samples.shape):
            sqrt_one_minus_alpha_cumprod = sqrt_one_minus_alpha_cumprod.unsqueeze(-1)

        noise = torch.randn(original_samples.shape, generator = self.generator, device = original_samples.device, dtype = original_samples.dtype)
        noisy_samples = (sqrt_alpha_cumprod * original_samples) + (sqrt_one_minus_alpha_cumprod) * noise
        return noisy_samples
    
    def _get_previous_timestep(self, timestep: int) -> int:
        prev_t = timestep - (self.num_training_steps // self.num_inference_steps)
        return prev_t
    
    def _get_variance(self, timestep: int) -> torch.Tensor:
        prev_t = self._get_previous_timestep(timestep)
        alpha_prod_t = self.alphas_cumprod[timestep]
        alpha_prod_prev_t = self.alphas_cumprod[prev_t] if prev_t >= 0 else self.one
        current_beta_t = 1 - alpha_prod_t / alpha_prod_prev_t

        variance = (1 - alpha_prod_prev_t) / (1 - alpha_prod_t) * current_beta_t
        variance = torch.clamp(variance, min = 1e-20)  # the variance should be greater than 0
        return variance
    
    def set_strength(self, strength = 1):
        start_step = self.num_inference_steps - int(self.num_inference_steps * strength)
        self.timesteps = self.timesteps[start_step:]
        self.start_step = start_step
        
    def step(self, timestep: int, latents: torch.Tensor, model_output: torch.Tensor):
        # model_output is the epsilon_theta, the predicted noise by the UNET
        t = timestep
        prev_t = self._get_previous_timestep(t)
        
        alpha_prod_t = self.alphas_cumprod[t]
        alpha_prod_prev_t = self.alphas_cumprod[prev_t] if prev_t >= 0 else self.one
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_prev_t
        current_alpha_t = alpha_prod_t / alpha_prod_prev_t
        current_beta_t = 1 - current_alpha_t

        # predicted x0, formula 15
        pred_original_sample = (latents - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5

        # formula 7 coeff
        pred_original_sample_coeff = (alpha_prod_prev_t ** 0.5 * current_beta_t) / beta_prod_t
        current_sample_coeff = (current_alpha_t ** 0.5 * beta_prod_t_prev) / beta_prod_t

        pred_prev_sample = pred_original_sample_coeff * pred_original_sample + current_sample_coeff * latents
        # if at last step, the variance is 0 since no need to remove noise
        variance = 0
        if t > 0:
            device = model_output.device
            noise = torch.randn(model_output.shape, generator = self.generator, device = device, dtype = model_output.dtype)
            variance = self._get_variance(t) ** 0.5 * noise
        
        pred_original_sample = pred_prev_sample + variance
        return pred_original_sample