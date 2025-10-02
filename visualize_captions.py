import json
import random
from pathlib import Path
import matplotlib.pyplot as plt
import cv2

# update path according to dir structure ......
BASE_DIR = Path("/kaggle/working/imgc") 

IMAGE_DIR = BASE_DIR / "data" / "Images"
CAPTIONS_FILE = BASE_DIR / "experiments" / "results" / "generated_captions.json"


if not IMAGE_DIR.exists():
    raise FileNotFoundError(f"❌ Images folder not found at {IMAGE_DIR}")
if not CAPTIONS_FILE.exists():
    raise FileNotFoundError(f"❌ Captions file not found at {CAPTIONS_FILE}")


with open(CAPTIONS_FILE, "r") as f:
    captions_dict = json.load(f)

# Pick random sample (5 images)
sample_images = random.sample(list(captions_dict.keys()), 5)


for img_name in sample_images:
    img_path = IMAGE_DIR / img_name
    caption = captions_dict[img_name]

    # Open with cv2
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"⚠️ Could not load {img_name}")
        continue

    # Convert BGR → RGB for matplotlib
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Plot
    plt.figure(figsize=(6, 4))
    plt.imshow(img_rgb)
    plt.title(f"{img_name}\nCaption: {caption}", fontsize=10)
    plt.axis("off")
    plt.show()