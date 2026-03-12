
import torchaudio.transforms as T
from PIL import Image
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
import numpy as np
import torch

def audio_spectrogram_image(waveform, power=2.0, sample_rate=48000, print=print, db=False, db_range=[35,120], justimage=False, log=False, figsize=(5, 4)):
    "Wrapper for calling above two routines at once, does Mel scale; Modified from PyTorch tutorial https://pytorch.org/tutorials/beginner/audio_feature_extractions_tutorial.html"
    melspec = mel_spectrogram(waveform, power=power, db=db, sample_rate=sample_rate, debug=log)
    melspec = melspec[0] # TODO: only left channel for now
    return spectrogram_image(melspec, title="MelSpectrogram", ylabel='mel bins (log freq)', db_range=db_range, justimage=justimage, figsize=figsize)

def spectrogram_image(
        spec, 
        title=None, 
        ylabel='freq_bin', 
        aspect='auto', 
        xmax=None, 
        db_range=[35,120], 
        justimage=False,
        figsize=(5, 4), # size of plot (if justimage==False)
    ):
    "Modified from PyTorch tutorial https://pytorch.org/tutorials/beginner/audio_feature_extractions_tutorial.html"
    fig = Figure(figsize=figsize, dpi=100) if not justimage else Figure(figsize=(4.145, 4.145), dpi=100, tight_layout=True)
    canvas = FigureCanvasAgg(fig)
    axs = fig.add_subplot()
    spec = spec.squeeze()
    im = axs.imshow(power_to_db(spec), origin='lower', aspect=aspect, vmin=db_range[0], vmax=db_range[1])
    if xmax:
        axs.set_xlim((0, xmax))
    if justimage:
        import matplotlib.pyplot as plt 
        axs.axis('off')
        plt.tight_layout()
    else: 
        axs.set_ylabel(ylabel)
        axs.set_xlabel('frame')
        axs.set_title(title or 'Spectrogram (dB)')
        fig.colorbar(im, ax=axs)
    canvas.draw()
    rgba = np.asarray(canvas.buffer_rgba())
    im = Image.fromarray(rgba)
    if justimage: # remove tiny white border
        b = 15 # border size 
        im = im.crop((b,b, im.size[0]-b, im.size[1]-b))
        #print(f"im.size = {im.size}")
    return im

def mel_spectrogram(waveform, power=2.0, sample_rate=48000, db=False, n_fft=1024, n_mels=128, debug=False):
    "calculates data array for mel spectrogram (in however many channels)"
    win_length = None
    hop_length = n_fft//2 # 512

    mel_spectrogram_op = T.MelSpectrogram(
        sample_rate=sample_rate, n_fft=n_fft, win_length=win_length, 
        hop_length=hop_length, center=True, pad_mode="reflect", power=power, 
        norm='slaney', onesided=True, n_mels=n_mels, mel_scale="htk")

    melspec = mel_spectrogram_op(waveform.float())
    if db: 
        amp_to_db_op = T.AmplitudeToDB()
        melspec = amp_to_db_op(melspec)
    if debug:
        #print_stats(melspec, print=print) 
        print(f"torch.max(melspec) = {torch.max(melspec)}")
        print(f"melspec.shape = {melspec.shape}")
    return melspec

def power_to_db(spec, *, amin = 1e-10):
    magnitude = np.asarray(spec)

    log_spec = 10.0 * np.log10(np.maximum(amin, magnitude))
    log_spec -= 10.0 * np.log10(np.maximum(amin, 1))

    log_spec = np.maximum(log_spec, log_spec.max() - 80)

    return log_spec