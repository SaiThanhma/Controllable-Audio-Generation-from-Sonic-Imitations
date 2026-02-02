import torch
import torchaudio
import torchcrepe
import math

import torchaudio
import torch
from frechet_audio_distance.utils import load_audio_task
from multiprocessing.dummy import Pool as ThreadPool
from tqdm import tqdm
import os
def load_audio_files(dir, sample_rate, channels = 1, audio_load_worker = 4, dtype="float32", verbose = True):
    task_results = []

    pool = ThreadPool(audio_load_worker)
    pbar = tqdm(total=len(os.listdir(dir)), disable=(not verbose))

    def update(*a):
        pbar.update()

    if verbose:
        print("Loading audio from {}...".format(dir))
    for fname in os.listdir(dir)[:10]:
        res = pool.apply_async(
            load_audio_task,
            args=(os.path.join(dir, fname), sample_rate, channels, dtype),
            callback=update,
        )
        task_results.append(res)
    pool.close()
    pool.join()

    return [k.get() for k in task_results]

audios = load_audio_files("/home/stud/dco/Desktop/sketch2sound/eval/generated", 48000, 1)
for k in audios:
    print(type(k))