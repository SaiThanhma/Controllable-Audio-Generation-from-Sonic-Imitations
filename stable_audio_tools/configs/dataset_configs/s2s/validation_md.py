import random
import os
def get_custom_metadata(info, audio):
    meta = info["original_data"]
    seconds_start = info["seconds_start"]
    seconds_total = info["seconds_total"]
    parts = []
    # ---- Extract possible metadata text ---- #

    # 1. title
    if "title" in meta:

        title = os.path.splitext(meta["title"])[0]
        parts.append(title)

    # 2. description
    if "description" in meta:
        parts.append(meta["description"])

    # 4. tags (list)
    if "tags" in meta:
        # shuffle tag list
        shuffled_tags = meta["tags"][:]
        random.shuffle(shuffled_tags)
        # append as separate entries
        for tag in shuffled_tags:
            parts.append(tag)

    # ---- Random subset of metadata ---- #

    k = random.randint(1, len(parts))    # select 1..N pieces
    selected = random.sample(parts, k)

    prompt = ", ".join(selected)

    # ---- Return final metadata ---- #
    return {
        "prompt": prompt,
        "seconds_start" : seconds_start,
        "seconds_total" : seconds_total,
        "control_signal": audio
    }
