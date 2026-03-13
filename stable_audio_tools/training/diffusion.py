import pytorch_lightning as pl
import gc
import random
import torch
import torchaudio
import typing as tp

import os
from tqdm import tqdm


from ema_pytorch import EMA
from einops import rearrange
from safetensors.torch import save_file
from torch import optim
from torch.nn import functional as F

from ..models.dit import ConditionedDiffusionModelWrapper
from .utils import create_optimizer_from_config, create_scheduler_from_config, log_audio, log_image, log_metric, load_audio
from ..inference.sampling import get_alphas_sigmas, sample
from pytorch_lightning.utilities.rank_zero import rank_zero_only
from ..eval import extract_csa, default_values
from torchcrepe.core import SAMPLE_RATE
from torchaudio import transforms as T
from transformers import ClapModel, ClapProcessor

from .vis import audio_spectrogram_image

class DiffusionCondTrainingWrapper(pl.LightningModule):
    '''
    Wrapper for training a conditional audio diffusion model.
    '''
    def __init__(
            self,
            model: ConditionedDiffusionModelWrapper,
            use_ema: bool,
            optimizer_configs: dict,
            cfg_dropout_prob : float,
            validation_timesteps: list #  = [0.1, 0.3, 0.5, 0.7, 0.9],
    ):
        super().__init__()

        self.diffusion = model

        if use_ema:
            self.diffusion_ema = EMA(
                self.diffusion.model,
                beta=0.9999,
                power=3/4,
                update_every=1,
                update_after_step=1,
                include_online_model=False
            )
        else:
            self.diffusion_ema = None

        self.cfg_dropout_prob = cfg_dropout_prob

        self.rng = torch.quasirandom.SobolEngine(1, scramble=True)

        self.optimizer_configs = optimizer_configs

        # Validation
        self.validation_timesteps = validation_timesteps

        self.validation_step_outputs = {}

        for validation_timestep in self.validation_timesteps:
            self.validation_step_outputs[f'val/loss_{validation_timestep:.1f}'] = []

    def configure_optimizers(self):
        diffusion_opt_config = self.optimizer_configs['diffusion']
        opt_diff = create_optimizer_from_config(diffusion_opt_config['optimizer'], self.diffusion.parameters())

        if "scheduler" in diffusion_opt_config:
            sched_diff = create_scheduler_from_config(diffusion_opt_config['scheduler'], opt_diff)
            sched_diff_config = {
                "scheduler": sched_diff,
                "interval": "step"
            }
            return [opt_diff], [sched_diff_config]

        return [opt_diff]

    # Precision is already handled by the trainer
    def training_step(self, batch, batch_idx):
        reals, metadata = batch

        reals = reals[0]

        loss_info = {}

        diffusion_input = reals

        loss_info["audio_reals"] = diffusion_input

        conditioning = self.diffusion.conditioner(metadata, self.device)
        
        with torch.no_grad():
            diffusion_input = self.diffusion.pretransform.encode(diffusion_input)

        # Draw uniformly distributed continuous timesteps
        t = self.rng.draw(reals.shape[0])[:, 0].to(self.device)
        alphas, sigmas = get_alphas_sigmas(t)

        # Combine the ground truth data and the noise
        alphas = alphas[:, None, None]
        sigmas = sigmas[:, None, None]
        noise = torch.randn_like(diffusion_input)

        # v-prediction
        noised_inputs = diffusion_input * alphas + noise * sigmas
        
        targets = noise * alphas - diffusion_input * sigmas

        #output = self.diffusion(noised_inputs, t, cond = conditioning, cfg_dropout_prob = self.cfg_dropout_prob, cfg_scale_text = 1, cfg_scale_controls = 1, control_signal=conditioning['control_signal'])
        output = self.diffusion(noised_inputs, t, cond=conditioning, cfg_dropout_prob = self.cfg_dropout_prob, cfg_scale = 1)

         # mse loss
        loss = ((output - targets)**2).mean()

        log_dict = {
            'train/mse_loss': loss.detach(),
            'train/std_data': diffusion_input.std(),
            'train/lr': self.trainer.optimizers[0].param_groups[0]['lr']
        }

        self.log_dict(log_dict, prog_bar=True, on_step=True)
        return loss

    def on_before_zero_grad(self, *args, **kwargs):
        if self.diffusion_ema is not None:
            self.diffusion_ema.update()

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):

        reals, metadata = batch

        reals = reals[0]

        diffusion_input = reals

        conditioning = self.diffusion.conditioner(metadata, self.device)

        diffusion_input = self.diffusion.pretransform.encode(diffusion_input)

        for validation_timestep in self.validation_timesteps:

            t = torch.full((reals.shape[0],), validation_timestep, device=self.device)

            # Calculate the noise schedule parameters for those timesteps
            alphas, sigmas = get_alphas_sigmas(t)

            # Combine the ground truth data and the noise
            alphas = alphas[:, None, None]
            sigmas = sigmas[:, None, None]
            noise = torch.randn_like(diffusion_input)
            noised_inputs = diffusion_input * alphas + noise * sigmas

            targets = noise * alphas - diffusion_input * sigmas

            output = self.diffusion(noised_inputs, t, cond=conditioning, cfg_dropout_prob = 0.0, cfg_scale_text = 1, cfg_scale_controls = 1, control_signal=conditioning['control_signal'])

            val_loss = ((output - targets) ** 2).mean()

            self.validation_step_outputs[f'val/loss_{validation_timestep:.1f}'].append(val_loss.item())

    def on_validation_epoch_end(self):

        for validation_timestep in self.validation_timesteps:
            outputs_key = f'val/loss_{validation_timestep:.1f}'
            val_loss = sum(self.validation_step_outputs[outputs_key]) / len(self.validation_step_outputs[outputs_key])

            # Gather losses across all GPUs
            val_loss = self.all_gather(val_loss).mean().item()

            log_metric(self.logger, outputs_key, val_loss, step=self.global_step)

        # Get average over all timesteps
        val_loss = torch.tensor([val for val in self.validation_step_outputs.values()]).mean()

        # Gather losses across all GPUs
        val_loss = self.all_gather(val_loss).mean().item()

        log_metric(self.logger, 'val/avg_loss', val_loss, step=self.global_step)

        # Reset validation losses
        for validation_timestep in self.validation_timesteps:
            self.validation_step_outputs[f'val/loss_{validation_timestep:.1f}'] = []


    def export_model(self, path, use_safetensors=False):
        if self.diffusion_ema is not None:
            self.diffusion.model = self.diffusion_ema.ema_model

        if use_safetensors:
            save_file(self.diffusion.state_dict(), path)
        else:
            torch.save({"state_dict": self.diffusion.state_dict()}, path)


class DiffusionCondDemoCallback(pl.Callback):
    def __init__(self,
                 demo_every,
                 sample_size,
                 demo_steps,
                 sample_rate,
                 demo_conditioning: tp.Optional[tp.Dict[str, tp.Any]],
                 demo_cfg_scale_text: tp.Optional[tp.List[int]],
                 demo_cfg_scale_controls : tp.Optional[tp.List[int]],
                 clap_ckpt_path : str,
                 outdir :str
    ):
        super().__init__()

        self.demo_every = demo_every
        self.demo_samples = sample_size
        self.demo_steps = demo_steps
        self.sample_rate = sample_rate
        self.last_demo_step = -1
        self.demo_conditioning = demo_conditioning
        self.clap_ckpt_path = clap_ckpt_path
        self.demo_cfg_scales_text = demo_cfg_scale_text
        self.demo_cfg_scales_controls = demo_cfg_scale_controls
        self.num_demos = len(self.demo_conditioning)
        self.outdir=outdir
        os.makedirs(outdir, exist_ok=True)
        model_name = "laion/clap-htsat-fused"

        self.clap_model = ClapModel.from_pretrained(model_name, device_map = 'cpu')
        self.clap_processor = ClapProcessor.from_pretrained(model_name)

        self.clap_model.eval()

        self.csa_resampler = None
        self.clap_resampler = None
        if self.sample_rate != SAMPLE_RATE:
            self.csa_resampler = T.Resample(self.sample_rate, SAMPLE_RATE)
        if self.sample_rate != 48000: # CLAP sr
            self.clap_resampler = T.Resample(self.sample_rate, 48000)

    @rank_zero_only
    @torch.no_grad()
    def on_train_batch_start(self, trainer, module: DiffusionCondTrainingWrapper, batch, batch_idx):
        if (trainer.global_step - 1) % self.demo_every != 0 or self.last_demo_step == trainer.global_step:
           return

        module.eval()

        print(f"Generating demo")
        self.last_demo_step = trainer.global_step
        demo_samples = self.demo_samples // module.diffusion.pretransform.downsampling_ratio

        noise = torch.randn([self.num_demos, module.diffusion.io_channels, demo_samples]).to(module.device)

        print("Getting conditioning")
        demo_cond_device = []
        for cond in self.demo_conditioning:
            cond_copy = dict(cond)
            cond_copy["control_signal"], cond_copy["seconds_start"], cond_copy["seconds_total"] = load_audio(
                cond_copy["control_signal"],
                sample_rate=self.sample_rate,
                sample_size=self.demo_samples,
                device = module.device
            )
            demo_cond_device.append(cond_copy)
        
        conditioning = module.diffusion.conditioner(demo_cond_device, module.device)

        cond_inputs = module.diffusion.get_conditioning_inputs(conditioning)

        if self.clap_resampler is not None:
            self.clap_resampler = self.clap_resampler.to(module.device)

        if self.csa_resampler is not None:
            self.csa_resampler = self.csa_resampler.to(module.device)

        self.clap_model = self.clap_model.to(module.device)

        for demo_cfg_scale_text, demo_cfg_scale_controls in zip(self.demo_cfg_scales_text, self.demo_cfg_scales_controls):

            print(f"Generating demo for cfg scale text {demo_cfg_scale_text} and cfg scale control {demo_cfg_scale_controls}")
            
            model = module.diffusion_ema.ema_model if module.diffusion_ema is not None else module.diffusion.model
            fakes = sample(model, noise, self.demo_steps, 0, **cond_inputs, cfg_dropout_prob = 0.0, cfg_scale_text=demo_cfg_scale_text, cfg_scale_controls=demo_cfg_scale_controls, control_signal = conditioning['control_signal'])
            fakes = module.diffusion.pretransform.decode(fakes)

            tag = f"step{trainer.global_step}_cfgT{demo_cfg_scale_text}_cfgC{demo_cfg_scale_controls}"
            fakes = fakes.float()

            for cond, fake in zip(demo_cond_device, fakes):
                safe_prompt = "".join(c if c.isalnum() else "_" for c in cond["prompt"])[:60]
                filename = f"{safe_prompt}_demo_{tag}_{trainer.global_step:08}.wav"

                #---------------------------------------------------------------------------
                # CLAP
                fake_clap = fake.mean(dim=-2, keepdim=False)
                if self.clap_resampler is not None:
                    fake_clap = self.clap_resampler(fake_clap)
                fake_clap = fake_clap.detach().float().cpu()
                clap_inputs = self.clap_processor(text=cond['prompt'], audio=fake_clap, return_tensors="pt", sampling_rate=48000).to(module.device)

                clap_outputs = self.clap_model(**clap_inputs)

                audio_embed = F.normalize(clap_outputs.audio_embeds, dim=-1)
                text_embeds = F.normalize(clap_outputs.text_embeds, dim=-1)

                score = (audio_embed * text_embeds).sum(-1)
                log_metric(
                    trainer.logger,
                    f"demo/{filename}/clap_score",
                    score.item(),
                )

                #---------------------------------------------------------------------------
                # CSA
                fake_csa = fake.mean(dim=-2, keepdim=True)
                control_signal = cond["control_signal"].squeeze(0).mean(0, keepdim = True).to(module.device)
                if self.csa_resampler is not None:
                    fake_csa = self.csa_resampler(fake_csa)
                    control_signal = self.csa_resampler(control_signal)
                T_min = min(control_signal.shape[-1], fake_csa.shape[-1])
                control_signal = control_signal[..., :T_min]
                fake_csa  = fake_csa[..., :T_min]

                loudness_err, centroid_err, pitch_err, chroma_err, periodicity_err = extract_csa(control_signal, fake_csa, SAMPLE_RATE, device = module.device, **default_values)
                log_metric(trainer.logger, f"demo/{filename}/loudness_err", loudness_err)
                log_metric(trainer.logger, f"demo/{filename}/centroid_err", centroid_err)
                log_metric(trainer.logger, f"demo/{filename}/pitch_err", pitch_err)
                log_metric(trainer.logger, f"demo/{filename}/chroma_err", chroma_err)
                log_metric(trainer.logger, f"demo/{filename}/periodicity_err", periodicity_err)

            max_per_sample = fakes.abs().amax(dim=(1, 2), keepdim=True)
            max_per_sample = torch.where(max_per_sample == 0, 
                                        torch.ones_like(max_per_sample), 
                                        max_per_sample)

            fakes = fakes / max_per_sample
            fakes = (fakes * 32767).to(torch.int16).cpu()
            fakes_out = rearrange(fakes, 'b d n -> d (b n)')
            
            filepath = os.path.join(self.outdir, f"demo_{tag}.wav")
            torchaudio.save(filepath, fakes_out, self.sample_rate)
            log_audio(trainer.logger, f"demo_{tag}", filepath, self.sample_rate)
            log_image(trainer.logger, f"demo_melspec_left_{tag}", audio_spectrogram_image(fakes_out))
            
        del fakes

        if self.clap_resampler is not None:
            self.clap_resampler = self.clap_resampler.to("cpu")
        if self.csa_resampler is not None:
            self.csa_resampler = self.csa_resampler.to("cpu")
        self.clap_model = self.clap_model.to('cpu')
        gc.collect()
        torch.cuda.empty_cache()
        module.train()