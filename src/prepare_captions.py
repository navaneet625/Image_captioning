import os
import json
from collections import defaultdict

def prepare_captions(captions_txt="data/captions.txt", output_json="data/captions.json"):
    """
    Converts captions.txt (image#idx \t caption) into a JSON:
      { "image.jpg": ["caption1", "caption2", ...], ... }
    """

    captions_dict = defaultdict(list)

    with open(captions_txt, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # split by first comma only
            if "," not in line:
                continue
            img_id_caption, caption = line.split(",",1)
            img_id = img_id_caption.split("#")[0]  # remove #0, #1 etc.
            captions_dict[img_id].append(caption)

    # save JSON
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(captions_dict, f, indent=2)

    print(f"✅ Saved captions.json for {len(captions_dict)} images at {output_json}")


if __name__ == "__main__":
    prepare_captions()
