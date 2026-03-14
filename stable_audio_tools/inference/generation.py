import numpy as np
import torch 
import typing as tp
from .sampling import sample_k
import torch
import torch
import numpy as np
import argparse

def save_activations(model, noise, conditioning_inputs, output_path):
    """
    Run a single forward pass through model.model (the U‑Net) and save all layer outputs.
    """
    model.model.eval()
    activations = {}

    def get_hook(name):
        def hook(module, input, output):
            if isinstance(output, torch.Tensor):
                activations[name] = output.detach().cpu()
            elif isinstance(output, (tuple, list)):
                activations[name] = [o.detach().cpu() if isinstance(o, torch.Tensor) else o for o in output]
            else:
                activations[name] = output
        return hook

    # Register hooks on every submodule of model.model
    hooks = []
    for name, module in model.model.named_modules():
        hook = module.register_forward_hook(get_hook(name))
        hooks.append(hook)

    # Forward pass (no gradients)
    with torch.no_grad():
        _ = model.model(noise, **conditioning_inputs, t = torch.full((noise.shape[0],), 0.5,device=noise.device), cfg_scale = 3, cfg_dropout_prob=0)

    # Remove hooks
    for hook in hooks:
        hook.remove()

    torch.save(activations, output_path)
    print(f"Activations saved to {output_path}")


def generate_diffusion_cond(
        model,
        steps: int = 250,
        cfg_scale_text=1.0,
        cfg_scale_controls=1.0,
        conditioning: dict = None,
        conditioning_tensors: tp.Optional[dict] = None,
        negative_conditioning: dict = None,
        negative_conditioning_tensors: tp.Optional[dict] = None,
        batch_size: int = 1,
        sample_size: int = 2097152,
        sample_rate: int = 48000,
        seed: int = -1,
        device: str = "cuda",
        init_audio: tp.Optional[tp.Tuple[int, torch.Tensor]] = None,
        init_noise_level: float = 1.0,
        return_latents = False,
        **sampler_kwargs
        ) -> torch.Tensor: 
    """
    Generate audio from a prompt using a diffusion model.
    
    Args:
        model: The diffusion model to use for generation.
        steps: The number of diffusion steps to use.
        cfg_scale: Classifier-free guidance scale 
        conditioning: A dictionary of conditioning parameters to use for generation.
        conditioning_tensors: A dictionary of precomputed conditioning tensors to use for generation.
        batch_size: The batch size to use for generation.
        sample_size: The length of the audio to generate, in samples.
        sample_rate: The sample rate of the audio to generate (Deprecated, now pulled from the model directly)
        seed: The random seed to use for generation, or -1 to use a random seed.
        device: The device to use for generation.
        init_audio: A tuple of (sample_rate, audio) to use as the initial audio for generation.
        init_noise_level: The noise level to use when generating from an initial audio sample.
        return_latents: Whether to return the latents used for generation instead of the decoded audio.
        **sampler_kwargs: Additional keyword arguments to pass to the sampler.    
    """

    # The length of the output in audio samples 
    audio_sample_size = sample_size



    # If this is latent diffusion, change sample_size instead to the downsampled latent size
    if model.pretransform is not None:
        sample_size = sample_size // model.pretransform.downsampling_ratio
        
    # Seed
    # The user can explicitly set the seed to deterministically generate the same output. Otherwise, use a random seed.
    seed = seed if seed != -1 else np.random.randint(0, 2**32 - 1)
    print(seed)
    torch.manual_seed(seed)
    # Define the initial noise immediately after setting the seed
    noise = torch.randn([batch_size, model.io_channels, sample_size], device=device)

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    torch.backends.cudnn.benchmark = False

    # Conditioning
    assert conditioning is not None or conditioning_tensors is not None, "Must provide either conditioning or conditioning_tensors"
    if conditioning_tensors is None:
        conditioning_tensors = model.conditioner(conditioning, device)
    conditioning_inputs = model.get_conditioning_inputs(conditioning_tensors)   

    model_dtype = next(model.model.parameters()).dtype
    noise = noise.type(model_dtype)
    conditioning_inputs = {k: v.type(model_dtype) if v is not None else v for k, v in conditioning_inputs.items()}
    # Now the generative AI part:
    # k-diffusion denoising process go!

    # k-diffusion denoising process go
    #sampled = sample_k(model.model, noise, init_audio, steps, **sampler_kwargs, **conditioning_inputs, cfg_scale_text = cfg_scale_text, cfg_scale_controls=cfg_scale_controls, batch_cfg=True, device=device, control_signal = conditioning_tensors['control_signal'][0])
    save_activations(model, noise, conditioning_inputs, 'model1_acts.pt')
    import sys; sys.exit()
    sampled = sample_k(model.model, noise, init_audio, steps, **sampler_kwargs, **conditioning_inputs, cfg_scale_text = cfg_scale_text, cfg_scale_controls=cfg_scale_controls, batch_cfg=True, device=device)
    # v-diffusion: 

    #sampled = sample(model.model, noise, steps, 0, **conditioning_tensors, embedding_scale=cfg_scale)
    del noise
    del conditioning_tensors
    del conditioning_inputs
    torch.cuda.empty_cache()
    # Denoising process done. 
    # If this is latent diffusion, decode latents back into audio
    if model.pretransform is not None and not return_latents:
        #cast sampled latents to pretransform dtype
        sampled = sampled.to(next(model.pretransform.parameters()).dtype)
        sampled = model.pretransform.decode(sampled)

    # Return audio
    return sampled