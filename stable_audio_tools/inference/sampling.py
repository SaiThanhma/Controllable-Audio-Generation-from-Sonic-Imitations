import torch
import math
from tqdm import trange, tqdm
import torch.distributions as dist

import k_diffusion as K

# Uses k-diffusion from https://github.com/crowsonkb/k-diffusion
# init_data is init_audio as latents (if this is latent diffusion)
# For sampling, init_data to none
# For variations, set init_data

def get_alphas_sigmas(t):
    """Returns the scaling factors for the clean image (alpha) and for the
    noise (sigma), given a timestep."""
    return torch.cos(t * math.pi / 2), torch.sin(t * math.pi / 2)

    
def sample_k(
        model_fn,
        noise,
        init_data=None,
        steps=100,
        sampler_type="dpmpp-2m-sde",
        sigma_min=0.01,
        sigma_max=100,
        rho=1.0,
        device="cuda",
        callback=None,
        **extra_args
    ):

    is_k_diff = sampler_type in ["k-heun", "k-lms", "k-dpmpp-2s-ancestral", "k-dpm-2", "k-dpm-fast", "k-dpm-adaptive", "dpmpp-2m-sde", "dpmpp-3m-sde","dpmpp-2m"]

    if is_k_diff:

        denoiser = K.external.VDenoiser(model_fn)

        # Make the list of sigmas. Sigma values are scalars related to the amount of noise each denoising step has
        sigmas = K.sampling.get_sigmas_polyexponential(steps, sigma_min, sigma_max, rho, device=device)
        # Scale the initial noise by sigma
        noise = noise * sigmas[0]

        if init_data is not None:
            # set the initial latent to the init_data, and noise it with initial sigma
            x = init_data + noise
        else:
            # SAMPLING
            # set the initial latent to noise
            x = noise

        if sampler_type == "k-heun":
            return K.sampling.sample_heun(denoiser, x, sigmas, disable=False, callback=callback, extra_args=extra_args)
        elif sampler_type == "k-lms":
            return K.sampling.sample_lms(denoiser, x, sigmas, disable=False, callback=callback, extra_args=extra_args)
        elif sampler_type == "k-dpmpp-2s-ancestral":
            return K.sampling.sample_dpmpp_2s_ancestral(denoiser, x, sigmas, disable=False, callback=callback, extra_args=extra_args)
        elif sampler_type == "k-dpm-2":
            return K.sampling.sample_dpm_2(denoiser, x, sigmas, disable=False, callback=callback, extra_args=extra_args)
        elif sampler_type == "k-dpm-fast":
            return K.sampling.sample_dpm_fast(denoiser, x, sigma_min, sigma_max, steps, disable=False, callback=callback, extra_args=extra_args)
        elif sampler_type == "k-dpm-adaptive":
            return K.sampling.sample_dpm_adaptive(denoiser, x, sigma_min, sigma_max, rtol=0.01, atol=0.01, disable=False, callback=callback, extra_args=extra_args)
        elif sampler_type == "dpmpp-2m":
            return K.sampling.sample_dpmpp_2m(denoiser, x, sigmas, disable=False, callback=callback, extra_args=extra_args)
        elif sampler_type == "dpmpp-2m-sde":
            return K.sampling.sample_dpmpp_2m_sde(denoiser, x, sigmas, disable=False, callback=callback, extra_args=extra_args)
        elif sampler_type == "dpmpp-3m-sde":
            return K.sampling.sample_dpmpp_3m_sde(denoiser, x, sigmas, disable=False, callback=callback, extra_args=extra_args)
        else:
            raise ValueError(f"Unknown sampler_type: {sampler_type}")
    else:
        raise ValueError(f"Unknown sampler type {sampler_type}")
        
@torch.no_grad()
def sample(model, x, steps, eta, callback=None, sigma_max=1.0, dist_shift=None, cfg_pp=False, **extra_args):
    """Draws samples from a model given starting noise. v-diffusion"""
    ts = x.new_ones([x.shape[0]])

    # Create the noise schedule
    t = torch.linspace(sigma_max, 0, steps + 1)[:-1]

    if dist_shift is not None:
        t = dist_shift.time_shift(t, x.shape[-1])

    alphas, sigmas = get_alphas_sigmas(t)

    # The sampling loop
    for i in trange(steps):

        if cfg_pp:
            # Get the model output (v, the predicted velocity)
            v, info = model(x, ts * t[i], return_info=True, **extra_args)

            if "uncond_output" in info:
                v_eps = info["uncond_output"]
            else:
                v_eps = v
        else:
            v = model(x, ts * t[i], **extra_args)
            v_eps = v

        # Predict the noise and the denoised data
        pred = x * alphas[i] - v * sigmas[i]
        eps = x * sigmas[i] + v_eps * alphas[i]

        # If we are not on the last timestep, compute the noisy data for the
        # next timestep.
        if i < steps - 1:
            # If eta > 0, adjust the scaling factor for the predicted noise
            # downward according to the amount of additional noise to add
            ddim_sigma = eta * (sigmas[i + 1]**2 / sigmas[i]**2).sqrt() * \
                (1 - alphas[i]**2 / alphas[i + 1]**2).sqrt()
            adjusted_sigma = (sigmas[i + 1]**2 - ddim_sigma**2).sqrt()

            # Recombine the predicted noise and predicted denoised data in the
            # correct proportions for the next step
            x = pred * alphas[i + 1] + eps * adjusted_sigma

            # Add the correct amount of fresh noise
            if eta:
                x += torch.randn_like(x) * ddim_sigma

        if callback is not None:
            denoised = pred
            callback({'x': x, 't': t[i], 'sigma': sigmas[i], 'i': i, 'denoised': denoised })

    # If we are on the last timestep, output the denoised data
    return pred