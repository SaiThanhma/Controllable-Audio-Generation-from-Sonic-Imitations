import torch
import torchcrepe
import math
#from utils import load_audio_files
from torchaudio import transforms as T
from torchcrepe.core import postprocess
import numpy as np
from torchcrepe.core import SAMPLE_RATE, WINDOW_SIZE

default_values = {
    "hop_length": SAMPLE_RATE//4, 
    "fmin": 50, 
    "fmax": 500, 
    "model_type": 'full', 
    "pitch_threshold": 0.5, 
    "loudness_threshold": -40
}

def loudness(audio, windows_size, hop_length, eps=1e-10):
    # Unfold into frames, compute RMS, then dB
    audio = torch.nn.functional.pad(
        audio,
        (windows_size // 2, windows_size // 2)
    )
    frames = audio.unfold(-1, windows_size, hop_length)  # (n_frames, frame_length)
    rms = torch.sqrt(torch.mean(frames**2, dim=-1))
    db = 20 * torch.log10(rms.clamp_min(eps))
    return db

def centroid(mag, freqs, eps=1e-10):
    # mag: torch.Size([1, 1025, 4459])
    # freqs: torch.Size([1025])
    freqs_view = freqs.view(1, -1, 1)
    num = torch.sum(freqs_view * mag, dim=-2)
    den = torch.sum(mag, dim=-2).clamp_min(eps)
    hz = (num / den)
    midi = hz_to_st(hz)
    return midi

def pitch_periodicity(audio, sr, hop_length, model_type, fmin, fmax, device):
    # audio: ([1, 2282615])
    audio = audio.to(device)
    with torch.no_grad():
        pitch, periodicity = torchcrepe.predict(
            audio=audio,
            sample_rate=sr,
            hop_length=hop_length,
            model=model_type,
            fmin=fmin, 
            fmax=fmax,
            device = device,
            return_periodicity=True,
            #decoder=torchcrepe.decode.viterbi,
            decoder=torchcrepe.decode.weighted_argmax
        )

    return pitch, periodicity


def extract_csa(audio_in, audio_gen, sr, hop_length, fmin, fmax, model_type, pitch_threshold, loudness_threshold, device):

    assert sr == SAMPLE_RATE
    assert audio_in.shape == audio_gen.shape

    spectrogram = T.Spectrogram(
        n_fft=WINDOW_SIZE,
        hop_length=hop_length,
        win_length=WINDOW_SIZE,
        power=1.0,
        center=True,
    ).to(device)

    audio_in = audio_in.to(device)
    audio_gen = audio_gen.to(device)
    loud_in = loudness(audio_in, WINDOW_SIZE, hop_length)
    loud_gen = loudness(audio_gen, WINDOW_SIZE, hop_length)

    freqs = torch.linspace(0, sr / 2, WINDOW_SIZE // 2 + 1, device=device)
    mag_in = spectrogram(audio_in)
    mag_gen = spectrogram(audio_gen)

    cent_in = centroid(mag_in, freqs)
    cent_gen = centroid(mag_gen, freqs)
    
    pitch_in, per_in = pitch_periodicity(audio=audio_in, sr=sr, hop_length=hop_length, model_type=model_type, fmin=fmin, fmax=fmax, device=device)
    pitch_gen, per_gen = pitch_periodicity(audio=audio_gen, sr=sr, hop_length=hop_length, model_type=model_type, fmin=fmin, fmax=fmax, device=device)

    nonsilent = loud_in > loudness_threshold
    voiced = (per_in > pitch_threshold) & (per_gen > pitch_threshold)
    pitch_mask = nonsilent & voiced
    #------------------------#------------------------#------------------------#------------------------#------------------------#------------------------#------------------------
    ctrl = audio_in.squeeze(0).cpu().squeeze().numpy()
    gen = audio_gen.squeeze(0).cpu().squeeze().numpy()

    # Cross-correlation
    correlation = np.correlate(ctrl, gen, mode='full')
    lag = np.argmax(correlation) - (len(gen) - 1)
    print(f"Cross-correlation lag (samples): {lag}")

    correlation2 = np.correlate(ctrl, ctrl, mode='full')
    lag2 = np.argmax(correlation2) - (len(ctrl) - 1)
    print(f"Cross-correlation lag2 (samples): {lag2}")

    import matplotlib.pyplot as plt
    plt.figure(figsize=(12,4))
    plt.plot(ctrl, label='control')
    plt.plot(gen, label='generated')
    plt.legend()
    plt.savefig(f'alignment_check {lag}.png')

    print(f"Total frames: {len(loud_in)}")
    print(f"Non‑silent frames: {nonsilent.sum().item()} (loudness > {loudness_threshold} dB)")
    print(f"Voiced input frames: {(per_in > pitch_threshold).sum().item()}")
    print(f"Voiced generated frames: {(per_gen > pitch_threshold).sum().item()}")
    print(f"Both voiced: {voiced.sum().item()}")
    print(f"Non‑silent & both voiced: {pitch_mask.sum().item()}")
    print(f"Loudness input: min={loud_in.min().item():.2f} dB, max={loud_in.max().item():.2f} dB")
    print(f"Periodicity input: min={per_in.min().item():.3f}, mean={per_in.mean().item():.3f}, max={per_in.max().item():.3f}")
    print(f"Periodicity generated: min={per_gen.min().item():.3f}, mean={per_gen.mean().item():.3f}, max={per_gen.max().item():.3f}")
    voiced_in = per_in > pitch_threshold
    if voiced_in.any():
        print(f"Pitch input (voiced frames): min={pitch_in[voiced_in].min().item():.1f} Hz, max={pitch_in[voiced_in].max().item():.1f} Hz")

    #------------------------#------------------------#------------------------#------------------------#------------------------#------------------------
    assert loud_in.shape == loud_gen.shape
    assert cent_in.shape == cent_gen.shape
    assert pitch_in.shape == pitch_gen.shape
    assert per_in.shape == per_gen.shape
    ref_shape = loud_in.shape
    assert cent_in.shape == ref_shape
    assert pitch_in.shape == ref_shape
    assert per_in.shape == ref_shape

    if nonsilent.sum() > 0:
        loudness_err = l1(loud_in[nonsilent], loud_gen[nonsilent]).item()
        centroid_err = l1(cent_in[nonsilent], cent_gen[nonsilent]).item()
    else:
        loudness_err = float('nan')
        centroid_err = float('nan')

    if pitch_mask.sum() > 0:
        pitch_in_st = hz_to_st(pitch_in[pitch_mask])
        pitch_gen_st = hz_to_st(pitch_gen[pitch_mask])

        pitch_err = l1(pitch_in_st, pitch_gen_st).item()

        chroma_err = chroma_distance(
            chroma_from_midi(pitch_in_st),
            chroma_from_midi(pitch_gen_st)
        ).mean().item()

        periodicity_err = l1(per_in[pitch_mask], per_gen[pitch_mask]).item()
    else:
        pitch_err = float('nan')
        chroma_err = float('nan')
        periodicity_err = float('nan')

    return loudness_err, centroid_err, pitch_err, chroma_err, periodicity_err


def l1(x, y):
    return torch.mean(torch.abs(x - y))

def chroma_from_midi(m):
    return torch.remainder(m, 12.0)

def chroma_distance(a, b):
    diff = torch.abs(a - b)
    return torch.minimum(diff, 12 - diff)

def hz_to_st(hz, eps=1e-10):
    hz = hz.clamp_min(eps)
    return 69 + 12 * torch.log2(hz / 440.0)
