import torchaudio
import os
from torchaudio import transforms as T
from tqdm import tqdm
from pathlib import Path
from multiprocessing.dummy import Pool as ThreadPool

def match_length(audio_ref, audio_to_fix):
    """
    Make audio_to_fix have the same length as audio_ref.
    Crops if too long, pads with zeros if too short.
    """
    target_len = audio_ref.shape[-1]
    cur_len = audio_to_fix.shape[-1]

    if cur_len > target_len:
        # crop
        audio_to_fix = audio_to_fix[..., :target_len]

    elif cur_len < target_len:
        # pad
        pad_amount = target_len - cur_len
        audio_to_fix = torch.nn.functional.pad(
            audio_to_fix, (0, pad_amount)
        )

    return audio_to_fix


def load_audio_task(f1, f2, sample_rate):

    assert Path(f1).stem == Path(f2).stem

    audio1, in_sr1 = torchaudio.load(f1)
    audio2, in_sr2 = torchaudio.load(f2)

    if in_sr1 != sample_rate:
        audio1 = T.Resample(in_sr1, sample_rate)(audio1)

    if in_sr2 != sample_rate:
        audio2 = T.Resample(in_sr2, sample_rate)(audio2)

    audio1 = audio1.mean(dim=0, keepdim=True)
    audio2 = audio2.mean(dim=0, keepdim=True)
    audio1 = audio1.unsqueeze(0)
    audio2 = audio2.unsqueeze(0)
    return audio1, audio2

def load_audio_files(dir1, dir2, sample_rate, audio_load_worker=4, verbose = True):
    task_results = []

    pool = ThreadPool(audio_load_worker)
    files1 = sorted(os.listdir(dir1))
    files2 = sorted(os.listdir(dir2))
    assert files1 == files2

    pbar = tqdm(total=len(files1), disable=not verbose)

    def update(*a):
        pbar.update()

    if verbose:
        print("Loading audio from {}...".format(dir1))
        print("Loading audio from {}...".format(dir2))

    for fname in files1[:10]:
        f1 = os.path.join(dir1, fname)
        f2 = os.path.join(dir2, fname)
        res = pool.apply_async(
            load_audio_task,
            args=(f1, f2, sample_rate),
            callback=update,
        )
        task_results.append(res)

    pool.close()
    pool.join()
    pbar.close()

    return [r.get() for r in task_results]