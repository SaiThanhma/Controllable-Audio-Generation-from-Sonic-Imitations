import random
import os
def get_custom_metadata(info, audio):
    """
    Build a randomized text prompt from available metadata.
    This implements the Stable Audio Open metadata strategy:
    - take metadata fields (title, description, tags, username, etc.)
    - randomly select a subset
    - shuffle the elements
    - randomize casing
    - sometimes prepend metadata keys
    - join into a single prompt string
    """
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

    # ---- Random casing transformations ---- #
    def random_case(s):
        r = random.random()
        if r < 0.7:
            return s
        elif r < 0.90:
            return s.upper()
        else:
            return s.lower()

    # ---- Join into final prompt ---- #
    selected = [random_case(s) for s in selected]

    prompt = ", ".join(selected)

    # ---- Return final metadata ---- #
    return {
        "prompt": prompt,
        "seconds_start" : seconds_start,
        "seconds_total" : seconds_total
        #"control_signal": audio
    }
