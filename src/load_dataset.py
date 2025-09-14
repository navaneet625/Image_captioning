import os
import json
from collections import defaultdict
from datasets import load_dataset

def prepare_flickr8k(output_dir="data"):
    """
    Downloads Flickr8k dataset from Hugging Face and prepares:
    - data/images/ (all images)
    - data/captions.json (mapping image -> captions)
    """

    print("📥 Downloading Flickr8k dataset from Hugging Face...")
    ds = load_dataset("atasoglu/flickr8k-dataset", data_dir="data")


    # Prepare output dirs
    images_dir = os.path.join(output_dir, "images")
    os.makedirs(images_dir, exist_ok=True)

    # Build captions dict
    captions_dict = defaultdict(list)

    for split in ["train", "validation", "test"]:
        for row in ds[split]:
            img = row["image"]   # PIL Image
            img_id = row["id"]   # filename
            caption = row["caption"]

            # Save image if not already saved
            img_path = os.path.join(images_dir, img_id)
            if not os.path.exists(img_path):
                img.save(img_path)

            # Collect caption
            captions_dict[img_id].append(caption)

    # Save captions.json
    captions_file = os.path.join(output_dir, "captions.json")
    with open(captions_file, "w") as f:
        json.dump(captions_dict, f, indent=2)

    print(f"✅ Saved {len(captions_dict)} images")
    print(f"📂 Images: {images_dir}")
    print(f"📜 Captions: {captions_file}")

if __name__ == "__main__":
    prepare_flickr8k()
