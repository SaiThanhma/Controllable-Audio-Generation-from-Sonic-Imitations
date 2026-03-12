import torch
from torch.nn import Parameter
from .diffusion import DiffusionCondDemoCallback

def create_training_wrapper_from_config(model_config, model):
    training_config = model_config["training"]
    from .diffusion import DiffusionCondTrainingWrapper
    return DiffusionCondTrainingWrapper(
        model, 
        use_ema = training_config["use_ema"],
        optimizer_configs=training_config["optimizer_configs"],
        validation_timesteps=training_config["validation_timesteps"],
        cfg_dropout_prob = training_config["cfg_dropout_prob"],
    )

def create_demo_callback_from_config(model_config):

    training_config = model_config.get('training')

    demo_config = training_config["demo"]
    return DiffusionCondDemoCallback(
        demo_every=demo_config["demo_every"], 
        sample_size=model_config["sample_size"],
        demo_conditioning=demo_config["demo_cond"],
        demo_steps=demo_config["demo_steps"], 
        sample_rate=model_config["sample_rate"],
        demo_cfg_scale_text=demo_config["demo_cfg_scales_text"],
        demo_cfg_scale_controls=demo_config["demo_cfg_scales_controls"],
        clap_ckpt_path=demo_config["clap_ckpt_path"],
        outdir=demo_config["outdir"],
    )

