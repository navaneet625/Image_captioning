import json
import random
from pathlib import Path
from PIL import Image
from IPython.display import display
import matplotlib.pyplot as plt
import cv2  # Import the OpenCV library

BASE_DIR = Path("/content/drive/MyDrive/image_caption")
IMAGE_DIR = BASE_DIR / "data" / "Images"
CAPTIONS_FILE = BASE_DIR / "results" / "generated_captions.json"

# Check if paths exist
if not IMAGE_DIR.exists():
    raise FileNotFoundError(f"Images folder not found at {IMAGE_DIR}")
if not CAPTIONS_FILE.exists():
    raise FileNotFoundError(f"Captions file not found at {CAPTIONS_FILE}")

# Load generated captions
with open(CAPTIONS_FILE, "r") as f:
    captions_dict = json.load(f)

# Pick 5 random images (change to 10 if needed)
sample_images = random.sample(list(captions_dict.keys()), 5)

# Display images with captions
for img_name in sample_images:
    img_path = IMAGE_DIR / img_name
    caption = captions_dict[img_name]

    try:
        # Use cv2 to open the image. It returns a NumPy array.
        img = cv2.imread(str(img_path))

        # Check if the image was loaded correctly
        if img is None:
            print(f"⚠️ Image {img_name} could not be loaded by OpenCV.")
            continue

        # Convert from BGR (OpenCV's default) to RGB (Matplotlib's default)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Use matplotlib to display the image
        plt.figure(figsize=(5, 3))
        plt.imshow(img_rgb)
        plt.title(f"{img_name}")
        plt.axis('off')  # Hide the axes
        plt.show()

        print(f"📷 {img_name}: {caption}\n")
    except FileNotFoundError:
        print(f"⚠️ Image {img_name} not found at {img_path}")
        