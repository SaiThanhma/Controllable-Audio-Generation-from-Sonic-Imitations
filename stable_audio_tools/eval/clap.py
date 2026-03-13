# from frechet_audio_distance import CLAPScore

# clap = CLAPScore(
#     submodel_name="630k-audioset",
#     enable_fusion=True,
#     verbose=True,
# )

# score = clap.score(
#     audio_dir="data/fad_eval_10k",
#     text_path="data/prompts.csv",
#     text_column="text",
# )

# print("Mean CLAP similarity:", score)

from transformers import ClapModel, ClapProcessor
import os
import requests
from tqdm import tqdm
import torch
import numpy as np
import torchaudio.transforms as T

# class CLAPScorer:

#     def __init__(self, texts, device="cuda"):

#         self.device = device

#         print("TEST")
#         self.model = laion_clap.CLAP_Module(
#             enable_fusion=True,
#             device=device
#         )

#         url = "https://huggingface.co/lukewys/laion_clap/resolve/main/630k-audioset-fusion-best.pt"
#         clap_path = "/home/stud/dco/storage/user/dco/clap_score/630k-audioset-fusion-best.pt"

#         if not os.path.exists(clap_path):
#             print("Downloading...")
#             os.makedirs(os.path.dirname(clap_path), exist_ok=True)

#             response = requests.get(url, stream=True)
#             total_size = int(response.headers.get("content-length", 0))

#             with open(clap_path, "wb") as file:
#                 with tqdm(total=total_size, unit="B", unit_scale=True) as progress_bar:
#                     for data in response.iter_content(chunk_size=8192):
#                         file.write(data)
#                         progress_bar.update(len(data))

#         self.model.load_ckpt(clap_path)
#         self.model.eval()

#         # ---- FIX STARTS HERE ----

#         ids = list(texts.keys())        # <-- missing
#         text_list = list(texts.values())  # <-- CLAP needs list[str]

#         self.text_emb = {}

#         with torch.no_grad():
#             emb = self.model.get_text_embedding(
#                 text_list,
#                 use_tensor=True
#             )

#         for i, e in zip(ids, emb):
#             self.text_emb[i] = e.to(device)

#         # ---- FIX ENDS HERE ----

#     @torch.no_grad()
#     def score(self, audio_batch, sample_rate, text_ids):
#         """
#         audio_batch: list[Tensor] OR Tensor (B, T)
#         text_ids: list[int]
#         """

#         # -----------------------------
#         # convert list -> tensor
#         # -----------------------------
#         if isinstance(audio_batch, list):
#             audio_batch = torch.stack(audio_batch, dim=0)

#         # ensure shape (B, T)
#         if audio_batch.dim() == 3:
#             audio_batch = audio_batch.squeeze(1)

#         # -----------------------------
#         # resample (once!)
#         # -----------------------------
#         if sample_rate != 48000:
#             resampler = T.Resample(sample_rate, 48000).to(audio_batch.device)
#             audio_batch = resampler(audio_batch)

#         # -----------------------------
#         # normalize per sample
#         # -----------------------------
#         audio_batch = audio_batch / (
#             audio_batch.abs().amax(dim=1, keepdim=True) + 1e-9
#         )

#         # -----------------------------
#         # CLAP forward (BATCHED)
#         # -----------------------------
#         audio_emb = self.model.get_audio_embedding_from_data(
#             x=audio_batch,
#             use_tensor=True
#         )

#         # collect text embeddings
#         text_emb = torch.stack(
#             [self.text_emb[i] for i in text_ids],
#             dim=0
#         )

#         scores = F.cosine_similarity(
#             audio_emb,
#             text_emb,
#             dim=1
#         )

#         return scores.cpu()


# Choose a text ID
# text_id = 0

# # Compute similarity score
# score = scorer.score(audio, sample_rate, [text_id])
# print("CLAP similarity score:", score)  # e.g., 0.123...

import torch
import torchaudio.transforms as T

text = ["balbalblablablablalbalbalb"]

sample_rate = 48000
duration = 3
#audio = [torch.randn(sample_rate * duration), torch.randn(sample_rate * duration)]  # mono, 48kHz

device = "cuda" if torch.cuda.is_available() else "cpu"
audios = torch.randn(sample_rate * duration, device = device)

model = ClapModel.from_pretrained("laion/clap-htsat-fused").to(device)
processor = ClapProcessor.from_pretrained("laion/clap-htsat-fused")
inputs = processor(text=text, audio=audios, return_tensors="pt", padding=True, sampling_rate=48000).to(device)
model.eval()

with torch.no_grad():
    outputs = model(**inputs)

# Cosine similarity (CLAP score)

print(outputs.audio_embeds.shape)
print(outputs.text_embeds.shape)
clap_score = torch.cosine_similarity(
    outputs.audio_embeds,
    outputs.text_embeds,
    dim = -1,
)
z1 = outputs.audio_embeds
z2 = outputs.text_embeds
s = (outputs.audio_embeds * outputs.text_embeds).sum(-1)

print("CLAP score:", clap_score)
# print("CLAP score:", s)
# z = outputs.text_embeds
# import torch
# import torch.nn.functional as F
# from transformers import ClapModel, ClapProcessor
# import torchaudio.transforms as T

# # ---------- Dummy context ----------
# class DummyModule:
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# class DummySelf:
#     def __init__(self):
#         self.sample_rate = 44100  # example different from 48k to test resampler creation
#         self.demo_conditioning = [
#             {"prompt": "balbalblablablablalbalbalb"},
#             #{"prompt": "rain falling on roof"},
#             #{"prompt": "a cat meowing"}
#         ]
#         self.module = DummyModule()

# # Instantiate dummy self
# self = DummySelf()
# from transformers import RobertaTokenizer, ClapFeatureExtractor
# # ---------- Your code snippet ----------
# self.clap_resampler = None
# if self.sample_rate != 48000:  # CLAP sr
#     self.clap_resampler = T.Resample(self.sample_rate, 48000).to(self.module.device)

# model_name = "laion/clap-htsat-fused"
# self.clap_model = ClapModel.from_pretrained(model_name).to(self.module.device)
# processor = ClapProcessor.from_pretrained("laion/clap-htsat-fused")

# # Extract prompts and compute text embeddings



# texts = [d["prompt"] for d in self.demo_conditioning]
# prompt_emb = {}
# tokenizer = RobertaTokenizer.from_pretrained(model_name)
# feature_extractor = ClapFeatureExtractor.from_pretrained(model_name)
# print(processor)
# audios_inputs = feature_extractor(audios, return_tensors="pt", padding=True, sampling_rate=48000).to(self.module.device)
# print(feature_extractor)
# i = 0
# for text in texts:

#     texts_input = tokenizer(text, return_tensors="pt").to(self.module.device)

#     with torch.inference_mode():
#         text_features = self.clap_model.get_text_features(
#             **texts_input
#         ).pooler_output
        
#         x = text_features
#         # Normalize and detach, store on CPU
#         prompt_emb[text] = F.normalize(text_features, dim=-1)
#         i += 1
#     with torch.inference_mode():
#         audio_features = self.clap_model.get_audio_features(**audios_inputs).pooler_output
#     print(torch.allclose(audio_features, z1))
#     print(torch.allclose(text_features, z2))
#     clap_score = torch.cosine_similarity(
#         audio_features,
#         text_features,
#         dim = -1,
#     )
#     print(clap_score)
# # ---------- Verification ----------
# print(f"Number of prompts: {len(prompt_emb)}")
# for text, emb in prompt_emb.items():
#     print(f"Text: {text[:30]}... -> embedding shape: {emb.shape}")  # Expected: torch.Size([1, 512])