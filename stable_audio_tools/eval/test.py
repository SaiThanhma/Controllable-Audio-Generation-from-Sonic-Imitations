import torchcrepe
import math
import librosa
import torch

import numpy as np

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


audio_ref = '/home/stud/dco/Desktop/sketch2sound/s2s_sounds/3/tests_assets_test.wav'
hop_length=256
audio_ref, fs = librosa.load(audio_ref, sr=None)
audio_ref = torch.from_numpy(audio_ref).unsqueeze(0)

# Get raw periodicity (before silence masking)
_, periodicity_raw = torchcrepe.predict(
    audio_ref, sample_rate=fs, hop_length=hop_length,
    fmin=50, fmax=550,
    model="full", return_periodicity=True, device=device
)
periodicity_raw = periodicity_raw.squeeze(0).cpu().numpy()

print("Raw periodicity - min:", periodicity_raw.min(), "max:", periodicity_raw.max(), "mean:", periodicity_raw.mean())
print("Frames with raw > 0.5:", (periodicity_raw > 0.5).sum())

import matplotlib.pyplot as plt

plt.figure(figsize=(10, 4))
plt.plot(audio_ref.squeeze().numpy())
plt.title("Waveform")
plt.xlabel("Sample")
plt.ylabel("Amplitude")
plt.tight_layout()

# Save the plot to a file (e.g., 'waveform.png')
plt.savefig('waveform.png', dpi=150, bbox_inches='tight')
plt.close()  # Close the figure to free memory


duration = 2.0
t = torch.linspace(0, duration, int(16000 * duration))
sine = 0.5 * torch.sin(2 * np.pi * 440 * t).unsqueeze(0)

_, periodicity_sine = torchcrepe.predict(
    sine, sample_rate=16000, hop_length=256,
    model="full", return_periodicity=True, device=device
)
periodicity_sine = periodicity_sine.squeeze(0).cpu().numpy()
print("Sine periodicity mean:", periodicity_sine.mean())

# import torch
# import torchcrepe
# import numpy as np
# import librosa

# fs = 16000
# hop_length = 256
# duration = 2.0  # seconds

# # Pure sine wave (fully periodic)
# t = np.linspace(0, duration, int(fs*duration), endpoint=False)
# sine = 0.5 * np.sin(2 * np.pi * 440 * t)  # 440 Hz

# # White noise (aperiodic)
# noise = np.random.normal(0, 0.5, len(t))

# # Mixed: first half sine, second half noise
# mixed = np.concatenate([sine[:len(t)//2], noise[len(t)//2:]])

# # Function to compute periodicity for a signal
# def get_periodicity(audio_np):
#     audio = torch.from_numpy(audio_np).float().unsqueeze(0).to(device)
#     _, periodicity = torchcrepe.predict(
#         audio, fs, hop_length=hop_length,
#         fmin=50, fmax=1500, model='full',
#         return_periodicity=True, device=device
#     )
#     periodicity = torchcrepe.threshold.Silence()(
#         periodicity, audio, fs, hop_length=hop_length
#     )
#     return periodicity.squeeze(0).cpu().numpy()

# period_sine = get_periodicity(sine)
# period_noise = get_periodicity(noise)
# period_mixed = get_periodicity(mixed)

# print("Sine mean periodicity:", period_sine.mean())
# print("Noise mean periodicity:", period_noise.mean())

# # For mixed, check that first half is high, second half low
# mid = len(period_mixed)//2
# print(period_mixed[:mid])
# print("Mixed first half mean:", period_mixed[:mid].mean())
# print("Mixed second half mean:", period_mixed[mid:].mean())