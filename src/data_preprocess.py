import json
from collections import defaultdict
from pathlib import Path
from sklearn.model_selection import train_test_split

def prepare_captions(captions_txt="data/captions.txt", output_json="data/captions.json"):
    """
    Converts captions.txt into JSON:
      { "image1.jpg": ["caption1", "caption2", ...], ... }
    """
    captions_dict = defaultdict(list)

    with open(captions_txt, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or "," not in line:
                continue            

            img_name, caption = line.split(",", 1)
            img_name = img_name.strip()

            if img_name.lower() in ["image", "image_id"]:
                continue

            captions_dict[img_name].append(caption.strip())

    counts = [len(v) for v in captions_dict.values()]
    avg_caps = sum(counts) / len(counts) if counts else 0
    print(f"📊 Average captions per image: {avg_caps:.2f}")

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(captions_dict, f, indent=2)

    print(f"✅ Saved {len(captions_dict)} images with captions at {output_json}")


def preprocess_dataset(images_dir: Path, captions_json: Path, processed_dir: Path, test_size: float = 0.2):
    """
    Split dataset into train/test JSON files only.
    """
    with open(captions_json, "r") as f:
        captions_data = json.load(f)

    available_images = {p.name for p in Path(images_dir).glob("*") 
                        if p.suffix.lower() in [".jpg", ".jpeg", ".png"]}

    fixed_captions = {img: caps for img, caps in captions_data.items() if img in available_images}

    dropped = len(captions_data) - len(fixed_captions)
    if dropped > 0:
        print(f"⚠️ Dropped {dropped} entries (no matching image file found).")

    all_images = list(fixed_captions.keys())
    if not all_images:
        raise ValueError("❌ No valid images found. Check filenames in captions.json and data/Images/")

    # Train/test split
    train_imgs, test_imgs = train_test_split(all_images, test_size=test_size, random_state=42)

    splits = {
        "train": {img: fixed_captions[img] for img in train_imgs},
        "test": {img: fixed_captions[img] for img in test_imgs},
    }

    processed_dir = Path(processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)

    for split, data in splits.items():
        with open(processed_dir / f"{split}.json", "w") as f:
            json.dump(data, f)

    print(f"✅ Preprocessing complete: {len(train_imgs)} train, {len(test_imgs)} test")
