import torch
from torch import nn
from torch.nn import functional as F
#from functools import partial
import numpy as np
import typing as tp

from .conditioners import MultiConditioner, create_multi_conditioner_from_conditioning_config
from .autoencoder import AudioAutoencoder
from .autoencoder import AutoencoderPretransform
from ..inference.generation import generate_diffusion_cond
from time import time

import typing as tp
import math
import torch

from einops import rearrange
from torch import nn
from torch.nn import functional as F

from .transformer import ContinuousTransformer

class FourierFeatures(nn.Module):
    def __init__(self, in_features, out_features, std=1.):
        super().__init__()
        assert out_features % 2 == 0
        self.weight = nn.Parameter(torch.randn(
            [out_features // 2, in_features]) * std)

    def forward(self, input):
        f = 2 * math.pi * input @ self.weight.T
        return torch.cat([f.cos(), f.sin()], dim=-1)

class DiffusionTransformer(nn.Module):
    def __init__(self, 
        io_channels=32, 
        patch_size=1,
        embed_dim=768,
        cond_token_dim=0,
        project_cond_tokens=True,
        global_cond_dim=0,
        project_global_cond=True,
        input_concat_dim=0,
        prepend_cond_dim=0,
        depth=12,
        num_heads=8,
        transformer_type: tp.Literal["continuous_transformer"] = "continuous_transformer",
        global_cond_type: tp.Literal["prepend", "adaLN"] = "prepend",
        timestep_cond_type: tp.Literal["global", "input_concat"] = "global",
        timestep_embed_dim=None,
        diffusion_objective: tp.Literal["v", "rectified_flow", "rf_denoiser"] = "v",
        **kwargs):

        super().__init__()
        
        self.cond_token_dim = cond_token_dim

        # Timestep embeddings
        self.timestep_cond_type = timestep_cond_type

        timestep_features_dim = 256

        self.timestep_features = FourierFeatures(1, timestep_features_dim)

        if timestep_cond_type == "global":
            timestep_embed_dim = embed_dim
        elif timestep_cond_type == "input_concat":
            assert timestep_embed_dim is not None, "timestep_embed_dim must be specified if timestep_cond_type is input_concat"
            input_concat_dim += timestep_embed_dim

        self.to_timestep_embed = nn.Sequential(
            nn.Linear(timestep_features_dim, timestep_embed_dim, bias=True),
            nn.SiLU(),
            nn.Linear(timestep_embed_dim, timestep_embed_dim, bias=True),
        )
        
        self.diffusion_objective = diffusion_objective

        if cond_token_dim > 0:
            # Conditioning tokens
            cond_embed_dim = cond_token_dim if not project_cond_tokens else embed_dim
            self.to_cond_embed = nn.Sequential(
                nn.Linear(cond_token_dim, cond_embed_dim, bias=False),
                nn.SiLU(),
                nn.Linear(cond_embed_dim, cond_embed_dim, bias=False)
            )
        else:
            cond_embed_dim = 0

        if global_cond_dim > 0:
            # Global conditioning
            global_embed_dim = global_cond_dim if not project_global_cond else embed_dim
            self.to_global_embed = nn.Sequential(
                nn.Linear(global_cond_dim, global_embed_dim, bias=False),
                nn.SiLU(),
                nn.Linear(global_embed_dim, global_embed_dim, bias=False)
            )

        if prepend_cond_dim > 0:
            # Prepend conditioning
            self.to_prepend_embed = nn.Sequential(
                nn.Linear(prepend_cond_dim, embed_dim, bias=False),
                nn.SiLU(),
                nn.Linear(embed_dim, embed_dim, bias=False)
            )

        self.input_concat_dim = input_concat_dim

        dim_in = io_channels + self.input_concat_dim

        self.patch_size = patch_size

        # Transformer

        self.transformer_type = transformer_type

        self.global_cond_type = global_cond_type

        if self.transformer_type == "continuous_transformer":

            global_dim = None

            if self.global_cond_type == "adaLN":
                # The global conditioning is projected to the embed_dim already at this point
                global_dim = embed_dim

            self.transformer = ContinuousTransformer(
                dim=embed_dim,
                depth=depth,
                dim_heads=embed_dim // num_heads,
                dim_in=dim_in * patch_size,
                dim_out=io_channels * patch_size,
                cross_attend = cond_token_dim > 0,
                cond_token_dim = cond_embed_dim,
                global_cond_dim=global_dim,
                **kwargs
            )
        else:
            raise ValueError(f"Unknown transformer type: {self.transformer_type}")

        self.preprocess_conv = nn.Conv1d(dim_in, dim_in, 1, bias=False)
        nn.init.zeros_(self.preprocess_conv.weight)
        self.postprocess_conv = nn.Conv1d(io_channels, io_channels, 1, bias=False)
        nn.init.zeros_(self.postprocess_conv.weight)

    def _forward(
        self, 
        x, 
        t, 
        mask=None,
        cross_attn_cond=None,
        cross_attn_cond_mask=None,
        input_concat_cond=None,
        global_embed=None,
        prepend_cond=None,
        prepend_cond_mask=None,
        return_info=False,
        exit_layer_ix=None,
        **kwargs):

        if cross_attn_cond is not None:
            cross_attn_cond = self.to_cond_embed(cross_attn_cond)

        if global_embed is not None:
            # Project the global conditioning to the embedding dimension
            global_embed = self.to_global_embed(global_embed)

        prepend_inputs = None 
        prepend_mask = None
        prepend_length = 0
        if prepend_cond is not None:
            # Project the prepend conditioning to the embedding dimension
            prepend_cond = self.to_prepend_embed(prepend_cond)

            prepend_inputs = prepend_cond
            if prepend_cond_mask is not None:
                prepend_mask = prepend_cond_mask

            prepend_length = prepend_cond.shape[1]

        if input_concat_cond is not None:
            # Interpolate input_concat_cond to the same length as x
            if input_concat_cond.shape[2] != x.shape[2]:
                input_concat_cond = F.interpolate(input_concat_cond, (x.shape[2], ), mode='nearest')

            x = torch.cat([x, input_concat_cond], dim=1)

        # Get the batch of timestep embeddings
        timestep_embed = self.to_timestep_embed(self.timestep_features(t[:, None])) # (b, embed_dim)

        # Timestep embedding is considered a global embedding. Add to the global conditioning if it exists

        if self.timestep_cond_type == "global":
            if global_embed is not None:
                global_embed = global_embed + timestep_embed
            else:
                global_embed = timestep_embed
        elif self.timestep_cond_type == "input_concat":
            x = torch.cat([x, timestep_embed.unsqueeze(1).expand(-1, -1, x.shape[2])], dim=1)

        # Add the global_embed to the prepend inputs if there is no global conditioning support in the transformer
        if self.global_cond_type == "prepend" and global_embed is not None:
            if prepend_inputs is None:
                # Prepend inputs are just the global embed, and the mask is all ones
                prepend_inputs = global_embed.unsqueeze(1)
                prepend_mask = torch.ones((x.shape[0], 1), device=x.device, dtype=torch.bool)
            else:
                # Prepend inputs are the prepend conditioning + the global embed
                prepend_inputs = torch.cat([prepend_inputs, global_embed.unsqueeze(1)], dim=1)
                prepend_mask = torch.cat([prepend_mask, torch.ones((x.shape[0], 1), device=x.device, dtype=torch.bool)], dim=1)

            prepend_length = prepend_inputs.shape[1]

        x = self.preprocess_conv(x) + x

        x = rearrange(x, "b c t -> b t c")

        extra_args = {}

        if self.global_cond_type == "adaLN":
            extra_args["global_cond"] = global_embed

        if self.patch_size > 1:
            x = rearrange(x, "b (t p) c -> b t (c p)", p=self.patch_size)

        if self.transformer_type == "continuous_transformer":
            # Masks not currently implemented for continuous transformer
            output = self.transformer(x, prepend_embeds=prepend_inputs, context=cross_attn_cond, return_info=return_info, exit_layer_ix=exit_layer_ix, **extra_args, **kwargs)

            if return_info:
                output, info = output

            # Avoid postprocessing on early exit
            if exit_layer_ix is not None:
                if return_info:
                    return output, info
                else:
                    return output

        output = rearrange(output, "b t c -> b c t")[:,:,prepend_length:]

        if self.patch_size > 1:
            output = rearrange(output, "b (c p) t -> b c (t p)", p=self.patch_size)

        output = self.postprocess_conv(output) + output

        if return_info:
            return output, info

        return output

    # def forward(
    #     self, 
    #     x, 
    #     t,
    #     cross_attn_cond=None,
    #     cross_attn_cond_mask=None,
    #     negative_cross_attn_cond=None, # Remove
    #     negative_cross_attn_mask=None, # Remove
    #     input_concat_cond=None,
    #     global_embed=None,
    #     negative_global_embed=None, # Remove
    #     prepend_cond=None, # Remove
    #     prepend_cond_mask=None, # Remove
    #     control_signal=None, # control_signal
    #     cfg_scale_text=1.0,
    #     cfg_scale_controls=1.0,
    #     cfg_dropout_prob=0.0,
    #     cfg_interval = (0, 1),
    #     causal=False,
    #     mask=None,
    #     return_info=False,
    #     exit_layer_ix=None,
    #     **kwargs):

    #     assert causal == False, "Causal mode is not supported for DiffusionTransformer"

    #     model_dtype = next(self.parameters()).dtype
        
    #     x = x.to(model_dtype)

    #     t = t.to(model_dtype)

    #     if cross_attn_cond is not None:
    #         cross_attn_cond = cross_attn_cond.to(model_dtype)

    #     if negative_cross_attn_cond is not None:
    #         negative_cross_attn_cond = negative_cross_attn_cond.to(model_dtype)

    #     if input_concat_cond is not None:
    #         input_concat_cond = input_concat_cond.to(model_dtype)

    #     if global_embed is not None:
    #         global_embed = global_embed.to(model_dtype)

    #     if negative_global_embed is not None:
    #         negative_global_embed = negative_global_embed.to(model_dtype)

    #     if prepend_cond is not None:
    #         prepend_cond = prepend_cond.to(model_dtype)

    #     if cross_attn_cond_mask is not None:
    #         cross_attn_cond_mask = cross_attn_cond_mask.bool()

    #         cross_attn_cond_mask = None # Temporarily disabling conditioning masks due to kernel issue for flash attention

    #     if prepend_cond_mask is not None:
    #         prepend_cond_mask = prepend_cond_mask.bool()

    #     sigma = torch.sin(t * math.pi / 2)
    #     alpha = torch.cos(t * math.pi / 2)

    #     # CFG dropout (Training)
    #     if cfg_dropout_prob > 0.0 and cfg_scale_text == 1.0 and cfg_scale_controls == 1.0:
    #         probs = torch.tensor([1.0 - (cfg_dropout_prob * 3), cfg_dropout_prob, cfg_dropout_prob, cfg_dropout_prob], device=control_signal.device)
    #         batch_size = x.shape[0]
    #         states = torch.multinomial(probs, batch_size, replacement=True) # (B,)
    #         drop_text_mask = (states == 1) | (states == 3)
    #         drop_control_mask = (states == 2) | (states == 3)

    #         null_text = torch.zeros_like(cross_attn_cond, device=cross_attn_cond.device)
    #         text_mask_expanded = drop_text_mask.view(-1, 1, 1)
    #         cross_attn_cond = torch.where(text_mask_expanded, null_text, cross_attn_cond)

    #         null_ctrl = torch.zeros_like(adding_cond, device=adding_cond.device)
    #         ctrl_mask_expanded = drop_control_mask.view(-1, 1, 1)
    #         adding_cond = torch.where(ctrl_mask_expanded, null_ctrl, adding_cond)
    #         noised_inputs = noised_inputs + control_signal

    #     if (cfg_scale_text != 1.0 or cfg_scale_controls != 1.0) and (cross_attn_cond is not None or prepend_cond is not None) and (cfg_interval[0] <= sigma[0] <= cfg_interval[1]):

    #         # Classifier-free guidance (Validation/Eval)
    #         # Concatenate conditioned and unconditioned inputs on the batch dimension

    #         x_full_cond = x + control_signal

    #         # Stack for batch processing: [full_cond, text_only, uncond]
    #         batch_inputs = torch.cat([x_full_cond, x, x], dim=0)
    #         batch_timestep = torch.cat([t, t, t], dim=0)

    #         batch_global_cond = torch.cat([global_embed, global_embed, global_embed], dim=0)
            
    #         # Handle CFG for cross-attention conditioning
    #         null_embed = torch.zeros_like(cross_attn_cond, device=cross_attn_cond.device)
    #         batch_cond = torch.cat([cross_attn_cond, cross_attn_cond, null_embed], dim=0)
            
    #         batch_output = self._forward(
    #             batch_inputs, 
    #             batch_timestep, 
    #             cross_attn_cond=batch_cond, 
    #             cross_attn_cond_mask=None, 
    #             mask = None, 
    #             input_concat_cond=None, 
    #             global_embed = batch_global_cond,
    #             prepend_cond = None,
    #             prepend_cond_mask = None,
    #             return_info = return_info,
    #             **kwargs)

    #         out_full, out_text, out_uncond = torch.chunk(batch_output, 3, dim=0)

    #         output = (
    #             out_uncond +
    #             cfg_scale_text * (out_text - out_uncond) +
    #             cfg_scale_controls * (out_full - out_text)
    #         )
    #         return output
            
    #     else:
    #         return self._forward(
    #             x,
    #             t,
    #             cross_attn_cond=cross_attn_cond, 
    #             cross_attn_cond_mask=cross_attn_cond_mask, 
    #             input_concat_cond=input_concat_cond, 
    #             global_embed=global_embed, 
    #             prepend_cond=prepend_cond, 
    #             prepend_cond_mask=prepend_cond_mask,
    #             mask=mask,
    #             return_info=return_info,
    #             **kwargs
    #         )

    def forward(
            self, 
            x, 
            t, 
            cross_attn_cond=None,
            cross_attn_cond_mask=None,
            negative_cross_attn_cond=None,
            negative_cross_attn_mask=None,
            input_concat_cond=None,
            global_embed=None,
            negative_global_embed=None,
            prepend_cond=None,
            prepend_cond_mask=None,
            cfg_scale=1.0,
            cfg_dropout_prob=0.0,
            cfg_interval = (0, 1),
            causal=False,
            scale_phi=0.0,
            mask=None,
            return_info=False,
            exit_layer_ix=None,
            **kwargs):

            assert causal == False, "Causal mode is not supported for DiffusionTransformer"

            model_dtype = next(self.parameters()).dtype
            
            x = x.to(model_dtype)

            t = t.to(model_dtype)

            if cross_attn_cond is not None:
                cross_attn_cond = cross_attn_cond.to(model_dtype)

            if negative_cross_attn_cond is not None:
                negative_cross_attn_cond = negative_cross_attn_cond.to(model_dtype)

            if input_concat_cond is not None:
                input_concat_cond = input_concat_cond.to(model_dtype)

            if global_embed is not None:
                global_embed = global_embed.to(model_dtype)

            if negative_global_embed is not None:
                negative_global_embed = negative_global_embed.to(model_dtype)

            if prepend_cond is not None:
                prepend_cond = prepend_cond.to(model_dtype)

            if cross_attn_cond_mask is not None:
                cross_attn_cond_mask = cross_attn_cond_mask.bool()

                cross_attn_cond_mask = None # Temporarily disabling conditioning masks due to kernel issue for flash attention

            if prepend_cond_mask is not None:
                prepend_cond_mask = prepend_cond_mask.bool()

            # Early exit bypasses CFG processing
            if exit_layer_ix is not None:
                assert self.transformer_type == "continuous_transformer", "exit_layer_ix is only supported for continuous_transformer"
                return self._forward(
                    x,
                    t,
                    cross_attn_cond=cross_attn_cond, 
                    cross_attn_cond_mask=cross_attn_cond_mask, 
                    input_concat_cond=input_concat_cond, 
                    global_embed=global_embed, 
                    prepend_cond=prepend_cond, 
                    prepend_cond_mask=prepend_cond_mask,
                    mask=mask,
                    return_info=return_info,
                    exit_layer_ix=exit_layer_ix,
                    **kwargs
                )

            # CFG dropout
            if cfg_dropout_prob > 0.0 and cfg_scale == 1.0:
                if cross_attn_cond is not None:
                    null_embed = torch.zeros_like(cross_attn_cond, device=cross_attn_cond.device)
                    dropout_mask = torch.bernoulli(torch.full((cross_attn_cond.shape[0], 1, 1), cfg_dropout_prob, device=cross_attn_cond.device)).to(torch.bool)
                    cross_attn_cond = torch.where(dropout_mask, null_embed, cross_attn_cond)

                if prepend_cond is not None:
                    null_embed = torch.zeros_like(prepend_cond, device=prepend_cond.device)
                    dropout_mask = torch.bernoulli(torch.full((prepend_cond.shape[0], 1, 1), cfg_dropout_prob, device=prepend_cond.device)).to(torch.bool)
                    prepend_cond = torch.where(dropout_mask, null_embed, prepend_cond)

            if self.diffusion_objective == "v":
                sigma = torch.sin(t * math.pi / 2)
                alpha = torch.cos(t * math.pi / 2)
            elif self.diffusion_objective in ["rectified_flow", "rf_denoiser"]:
                sigma = t

            if cfg_scale != 1.0 and (cross_attn_cond is not None or prepend_cond is not None) and (cfg_interval[0] <= sigma[0] <= cfg_interval[1]):

                # Classifier-free guidance
                # Concatenate conditioned and unconditioned inputs on the batch dimension            
                batch_inputs = torch.cat([x, x], dim=0)
                batch_timestep = torch.cat([t, t], dim=0)

                if global_embed is not None:
                    batch_global_cond = torch.cat([global_embed, global_embed], dim=0)
                else:
                    batch_global_cond = None

                if input_concat_cond is not None:
                    batch_input_concat_cond = torch.cat([input_concat_cond, input_concat_cond], dim=0)
                else:
                    batch_input_concat_cond = None

                batch_cond = None
                batch_cond_masks = None
                
                # Handle CFG for cross-attention conditioning
                if cross_attn_cond is not None:

                    null_embed = torch.zeros_like(cross_attn_cond, device=cross_attn_cond.device)

                    # For negative cross-attention conditioning, replace the null embed with the negative cross-attention conditioning
                    if negative_cross_attn_cond is not None:

                        # If there's a negative cross-attention mask, set the masked tokens to the null embed
                        if negative_cross_attn_mask is not None:
                            negative_cross_attn_mask = negative_cross_attn_mask.to(torch.bool).unsqueeze(2)

                            negative_cross_attn_cond = torch.where(negative_cross_attn_mask, negative_cross_attn_cond, null_embed)
                        
                        batch_cond = torch.cat([cross_attn_cond, negative_cross_attn_cond], dim=0)

                    else:
                        batch_cond = torch.cat([cross_attn_cond, null_embed], dim=0)

                    if cross_attn_cond_mask is not None:
                        batch_cond_masks = torch.cat([cross_attn_cond_mask, cross_attn_cond_mask], dim=0)
                
                batch_prepend_cond = None
                batch_prepend_cond_mask = None

                if prepend_cond is not None:

                    null_embed = torch.zeros_like(prepend_cond, device=prepend_cond.device)

                    batch_prepend_cond = torch.cat([prepend_cond, null_embed], dim=0)
                            
                    if prepend_cond_mask is not None:
                        batch_prepend_cond_mask = torch.cat([prepend_cond_mask, prepend_cond_mask], dim=0)
            

                if mask is not None:
                    batch_masks = torch.cat([mask, mask], dim=0)
                else:
                    batch_masks = None
                
                batch_output = self._forward(
                    batch_inputs, 
                    batch_timestep, 
                    cross_attn_cond=batch_cond, 
                    cross_attn_cond_mask=batch_cond_masks, 
                    mask = batch_masks, 
                    input_concat_cond=batch_input_concat_cond, 
                    global_embed = batch_global_cond,
                    prepend_cond = batch_prepend_cond,
                    prepend_cond_mask = batch_prepend_cond_mask,
                    return_info = return_info,
                    **kwargs)

                if return_info:
                    batch_output, info = batch_output

                cond_output, uncond_output = torch.chunk(batch_output, 2, dim=0)

                cfg_output = uncond_output + (cond_output - uncond_output) * cfg_scale

                # CFG Rescale
                if scale_phi != 0.0:
                    cond_out_std = cond_output.std(dim=1, keepdim=True)
                    out_cfg_std = cfg_output.std(dim=1, keepdim=True)
                    output = scale_phi * (cfg_output * (cond_out_std/out_cfg_std)) + (1-scale_phi) * cfg_output
                else:
                    output = cfg_output
            
                if return_info:
                    info["uncond_output"] = uncond_output
                    return output, info

                return output
                
            else:
                return self._forward(
                    x,
                    t,
                    cross_attn_cond=cross_attn_cond, 
                    cross_attn_cond_mask=cross_attn_cond_mask, 
                    input_concat_cond=input_concat_cond, 
                    global_embed=global_embed, 
                    prepend_cond=prepend_cond, 
                    prepend_cond_mask=prepend_cond_mask,
                    mask=mask,
                    return_info=return_info,
                    **kwargs
                )

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

    # def forward(self,
    #             x,
    #             t,
    #             control_signal,
    #             cross_attn_cond,
    #             cross_attn_mask,
    #             input_concat_cond,
    #             global_cond,
    #             prepend_cond,
    #             prepend_cond_mask,
    #             cfg_scale_text : int,
    #             cfg_scale_controls : int,
    #             cfg_dropout_prob: float,
    #             ):
    def forward(self,
                x,
                t,
                cross_attn_cond,
                cross_attn_mask,
                input_concat_cond,
                global_cond,
                prepend_cond,
                cfg_scale,
                prepend_cond_mask,
                cfg_dropout_prob: float,
                batch_cfg = True,
                cfg_scale_text : int = 1,
                cfg_scale_controls : int = 1,
                ):
        # return self.model(
        #     x,
        #     t,
        #     control_signal=control_signal,
        #     cross_attn_cond=cross_attn_cond,
        #     cross_attn_cond_mask=cross_attn_mask,
        #     input_concat_cond=input_concat_cond,
        #     prepend_cond=prepend_cond,
        #     prepend_cond_mask=prepend_cond_mask,
        #     cfg_scale_text=cfg_scale_text,
        #     cfg_scale_controls=cfg_scale_controls,
        #     cfg_dropout_prob=cfg_dropout_prob,
        #     global_embed=global_cond,
        #     )
        return self.model(
            x,
            t,
            cross_attn_cond=cross_attn_cond,
            cross_attn_cond_mask=cross_attn_mask,
            input_concat_cond=input_concat_cond,
            prepend_cond=prepend_cond,
            prepend_cond_mask=prepend_cond_mask,
            #cfg_scale_text=cfg_scale_text,
            #cfg_scale_controls=cfg_scale_controls,
            cfg_scale = cfg_scale,
            cfg_dropout_prob = cfg_dropout_prob,
            global_embed=global_cond,
            )

class ConditionedDiffusionModelWrapper(nn.Module):
    """
    A diffusion model that takes in conditioning
    """
    def __init__(self, config: tp.Dict[str, tp.Any]):
        super().__init__()

        model_config = config["model"]
        diffusion_config = model_config["diffusion"]
        diffusion_model_config = diffusion_config["config"]

        sample_rate = config["sample_rate"]
        self.model = DiTWrapper(
            io_channels=diffusion_model_config["io_channels"],
            embed_dim=diffusion_model_config["embed_dim"],
            depth=diffusion_model_config["depth"],
            num_heads=diffusion_model_config["num_heads"],
            cond_token_dim=diffusion_model_config["cond_token_dim"],
            global_cond_dim=diffusion_model_config["global_cond_dim"],
            project_cond_tokens=diffusion_model_config["project_cond_tokens"],
            transformer_type=diffusion_model_config["transformer_type"],
        )
        self.io_channels = model_config["io_channels"]
        self.sample_rate = sample_rate

        self.cross_attn_cond_ids = diffusion_config["cross_attention_cond_ids"]
        self.global_cond_ids = diffusion_config["global_cond_ids"]
        self.input_concat_ids = diffusion_config["input_concat_ids"]
        self.prepend_cond_ids = diffusion_config["prepend_cond_ids"]
        pretransform_config = model_config.get("pretransform")

        autoencoder_config = {
            "sample_rate": sample_rate,
            "model": pretransform_config["config"],
        }

        autoencoder = AudioAutoencoder(autoencoder_config)

        iterate_batch = pretransform_config["iterate_batch"]

        self.pretransform = AutoencoderPretransform(
            autoencoder,
            iterate_batch=iterate_batch
        )

        conditioning_config = model_config["conditioning"]
        self.conditioner = create_multi_conditioner_from_conditioning_config(
            conditioning_config,
            pretransform=self.pretransform
        )
        self.min_input_length = self.pretransform.downsampling_ratio
        self.min_input_length *= self.model.model.patch_size
    # def __init__(
    #         self,
    #         model: DiTWrapper,
    #         conditioner: MultiConditioner,
    #         io_channels,
    #         sample_rate,
    #         min_input_length: int,
    #         pretransform: tp.Optional[Pretransform],
    #         cross_attn_cond_ids: tp.List[str],
    #         global_cond_ids: tp.List[str],
    #         input_concat_ids: tp.List[str],
    #         prepend_cond_ids: tp.List[str],
    #         ):
    #     super().__init__()

    #     self.model = model
    #     self.conditioner = conditioner
    #     self.io_channels = io_channels
    #     self.sample_rate = sample_rate
    #     self.pretransform = pretransform
    #     self.cross_attn_cond_ids = cross_attn_cond_ids
    #     self.global_cond_ids = global_cond_ids
    #     self.input_concat_ids = input_concat_ids
    #     self.prepend_cond_ids = prepend_cond_ids
    #     self.min_input_length = min_input_length

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

    # def forward(self, x: torch.Tensor, t: torch.Tensor, cond: tp.Dict[str, tp.Any], cfg_dropout_prob, cfg_scale_text, cfg_scale_controls, control_signal):
    #     return self.model(x, t, **self.get_conditioning_inputs(cond), cfg_dropout_prob = cfg_dropout_prob, cfg_scale_text = cfg_scale_text, cfg_scale_controls = cfg_scale_controls, control_signal= control_signal)
    def forward(self, x: torch.Tensor, t: torch.Tensor, cond: tp.Dict[str, tp.Any], **kwargs):
        return self.model(x, t, **self.get_conditioning_inputs(cond), **kwargs)
    
    def generate(self, *args, **kwargs):
        return generate_diffusion_cond(self, *args, **kwargs)