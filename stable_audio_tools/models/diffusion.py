import torch
from torch import nn
from torch.nn import functional as F
#from functools import partial
import numpy as np
import typing as tp
import random

from .conditioners import MultiConditioner, create_multi_conditioner_from_conditioning_config
from .dit import DiffusionTransformer
from .factory import create_pretransform_from_config
from .pretransforms import Pretransform
from ..inference.generation import generate_diffusion_cond

from time import time

class Profiler:

    def __init__(self):
        self.ticks = [[time(), None]]

    def tick(self, msg):
        self.ticks.append([time(), msg])

    def __repr__(self):
        rep = 80 * "=" + "\n"
        for i in range(1, len(self.ticks)):
            msg = self.ticks[i][1]
            ellapsed = self.ticks[i][0] - self.ticks[i - 1][0]
            rep += msg + f": {ellapsed*1000:.2f}ms\n"
        rep += 80 * "=" + "\n\n\n"
        return rep


class DiTWrapper(nn.Module): 
    def __init__(
        self,
        io_channels: int,
        embed_dim: int,
        depth: int,
        num_heads: int,
        cond_token_dim: int,
        global_cond_dim: int,
        project_cond_tokens: bool,
        transformer_type: str,
    ):
        super().__init__()
        self.model = DiffusionTransformer(
            io_channels=io_channels,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            cond_token_dim=cond_token_dim,
            global_cond_dim=global_cond_dim,
            project_cond_tokens=project_cond_tokens,
            transformer_type=transformer_type,
        )

    def forward(self,
                x,
                t,
                control_signal,
                cross_attn_cond,
                cross_attn_mask,
                input_concat_cond,
                global_cond,
                prepend_cond,
                prepend_cond_mask,
                cfg_scale_text : int,
                cfg_scale_controls : int,
                cfg_dropout_prob: float,
                ):

        return self.model(
            x,
            t,
            control_signal=control_signal,
            cross_attn_cond=cross_attn_cond,
            cross_attn_cond_mask=cross_attn_mask,
            input_concat_cond=input_concat_cond,
            prepend_cond=prepend_cond,
            prepend_cond_mask=prepend_cond_mask,
            cfg_scale_text=cfg_scale_text,
            cfg_scale_controls=cfg_scale_controls,
            cfg_dropout_prob=cfg_dropout_prob,
            global_embed=global_cond,
            )

class ConditionedDiffusionModelWrapper(nn.Module):
    """
    A diffusion model that takes in conditioning
    """
    def __init__(
            self,
            model: DiTWrapper,
            conditioner: MultiConditioner,
            io_channels,
            sample_rate,
            min_input_length: int,
            pretransform: tp.Optional[Pretransform],
            cross_attn_cond_ids: tp.List[str],
            global_cond_ids: tp.List[str],
            input_concat_ids: tp.List[str],
            prepend_cond_ids: tp.List[str],
            ):
        super().__init__()

        self.model = model
        self.conditioner = conditioner
        self.io_channels = io_channels
        self.sample_rate = sample_rate
        self.pretransform = pretransform
        self.cross_attn_cond_ids = cross_attn_cond_ids
        self.global_cond_ids = global_cond_ids
        self.input_concat_ids = input_concat_ids
        self.prepend_cond_ids = prepend_cond_ids
        self.min_input_length = min_input_length

    def get_conditioning_inputs(self, conditioning_tensors: tp.Dict[str, tp.Any]):
        cross_attention_input = None
        cross_attention_masks = None
        global_cond = None
        input_concat_cond = None
        prepend_cond = None
        prepend_cond_mask = None

        if len(self.cross_attn_cond_ids) > 0:
            # Concatenate all cross-attention inputs over the sequence dimension
            # Assumes that the cross-attention inputs are of shape (batch, seq, channels)
            cross_attention_input = []
            cross_attention_masks = []

            for key in self.cross_attn_cond_ids:
                cross_attn_in, cross_attn_mask = conditioning_tensors[key]

                # Add sequence dimension if it's not there
                if len(cross_attn_in.shape) == 2:
                    cross_attn_in = cross_attn_in.unsqueeze(1)
                    cross_attn_mask = cross_attn_mask.unsqueeze(1)

                cross_attention_input.append(cross_attn_in)
                cross_attention_masks.append(cross_attn_mask)

            cross_attention_input = torch.cat(cross_attention_input, dim=1)
            cross_attention_masks = torch.cat(cross_attention_masks, dim=1)

        if len(self.global_cond_ids) > 0:
            # Concatenate all global conditioning inputs over the channel dimension
            # Assumes that the global conditioning inputs are of shape (batch, channels)
            global_conds = []
            for key in self.global_cond_ids:
                global_cond_input = conditioning_tensors[key][0]

                global_conds.append(global_cond_input)

            # Concatenate over the channel dimension
            global_cond = torch.cat(global_conds, dim=-1)

            if len(global_cond.shape) == 3:
                global_cond = global_cond.squeeze(1)

        if len(self.input_concat_ids) > 0:
            # Concatenate all input concat conditioning inputs over the channel dimension
            # Assumes that the input concat conditioning inputs are of shape (batch, channels, seq)
            input_concat_cond = torch.cat([conditioning_tensors[key][0] for key in self.input_concat_ids], dim=1)

        if len(self.prepend_cond_ids) > 0:
            # Concatenate all prepend conditioning inputs over the sequence dimension
            # Assumes that the prepend conditioning inputs are of shape (batch, seq, channels)
            prepend_conds = []
            prepend_cond_masks = []

            for key in self.prepend_cond_ids:
                prepend_cond_input, prepend_cond_mask = conditioning_tensors[key]
                prepend_conds.append(prepend_cond_input)
                prepend_cond_masks.append(prepend_cond_mask)

            prepend_cond = torch.cat(prepend_conds, dim=1)
            prepend_cond_mask = torch.cat(prepend_cond_masks, dim=1)

        return {
            "cross_attn_cond": cross_attention_input,
            "cross_attn_mask": cross_attention_masks,
            "global_cond": global_cond,
            "input_concat_cond": input_concat_cond,
            "prepend_cond": prepend_cond,
            "prepend_cond_mask": prepend_cond_mask,
        }

    def forward(self, x: torch.Tensor, t: torch.Tensor, cond: tp.Dict[str, tp.Any], cfg_dropout_prob, cfg_scale_text, cfg_scale_controls, control_signal):
        return self.model(x, t, **self.get_conditioning_inputs(cond), cfg_dropout_prob = cfg_dropout_prob, cfg_scale_text = cfg_scale_text, cfg_scale_controls = cfg_scale_controls, control_signal= control_signal)

    def generate(self, *args, **kwargs):
        return generate_diffusion_cond(self, *args, **kwargs)


def create_diffusion_cond_from_config(config: tp.Dict[str, tp.Any]):

    model_config = config["model"]

    diffusion_config = model_config['diffusion']

    diffusion_model_config = diffusion_config['config']

    diffusion_model = DiTWrapper(
        io_channels=diffusion_model_config["io_channels"],
        embed_dim=diffusion_model_config["embed_dim"],
        depth=diffusion_model_config["depth"],
        num_heads=diffusion_model_config["num_heads"],
        cond_token_dim=diffusion_model_config["cond_token_dim"],
        global_cond_dim=diffusion_model_config["global_cond_dim"],
        project_cond_tokens=diffusion_model_config["project_cond_tokens"],
        transformer_type=diffusion_model_config["transformer_type"],
    )

    io_channels = model_config['io_channels']

    sample_rate = config['sample_rate']

    cross_attention_ids = diffusion_config['cross_attention_cond_ids']
    global_cond_ids = diffusion_config['global_cond_ids']
    input_concat_ids = diffusion_config['input_concat_ids']
    prepend_cond_ids = diffusion_config['prepend_cond_ids']

    pretransform = model_config.get("pretransform")

    pretransform = create_pretransform_from_config(pretransform, sample_rate)
    min_input_length = pretransform.downsampling_ratio

    conditioning_config = model_config['conditioning']
    conditioner = create_multi_conditioner_from_conditioning_config(conditioning_config, pretransform=pretransform)
    min_input_length *= diffusion_model.model.patch_size
    
    return ConditionedDiffusionModelWrapper(
        diffusion_model,
        conditioner,
        min_input_length=min_input_length,
        sample_rate=sample_rate,
        cross_attn_cond_ids=cross_attention_ids,
        global_cond_ids=global_cond_ids,
        input_concat_ids=input_concat_ids,
        prepend_cond_ids=prepend_cond_ids,
        pretransform=pretransform,
        io_channels=io_channels,
    )