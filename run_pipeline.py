# run_pipeline.py
import os
import shutil
from pathlib import Path
import subprocess
import torch
import json
from sklearn.model_selection import train_test_split
from src.prepare_captions import prepare_captions

# ----------------------------
# ⚡ Config
# ----------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
IMAGES_DIR = DATA_DIR / "Images"
CAPTIONS_JSON = DATA_DIR / "captions.json"
CAPTIONS_TXT = DATA_DIR / "captions.txt"
PROC_DIR = DATA_DIR / "processed"

EXPERIMENTS_DIR = BASE_DIR / "experiments"
CHECKPOINT_DIR = EXPERIMENTS_DIR / "checkpoints"
RESULTS_DIR = EXPERIMENTS_DIR / "results"

# ----------------------------
# ⚡ Remove old checkpoints to rebuild vocab
# ----------------------------
import shutil
shutil.rmtree(CHECKPOINT_DIR, ignore_errors=True)

CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ----------------------------
# 1️⃣ Cleanup functions
# ----------------------------
def remove_pycache_and_pyc(root="."):
    for dirpath, dirnames, filenames in os.walk(root):
        if "__pycache__" in dirnames:
            shutil.rmtree(os.path.join(dirpath, "__pycache__"))
        for file in filenames:
            if file.endswith(".pyc"):
                os.remove(os.path.join(dirpath, file))

def remove_experiment_data(root="experiments"):
    root_path = Path(root)
    if root_path.exists():
        shutil.rmtree(root_path)
    root_path.mkdir(parents=True, exist_ok=True)

def remove_temp_files(root="."):
    patterns = ["*.tmp", "*.log"]
    for pattern in patterns:
        for file_path in Path(root).rglob(pattern):
            file_path.unlink()

def remove_dataset_artifacts():
    """Clean old dataset artifacts (captions.json + processed splits)."""
    if CAPTIONS_JSON.exists():
        CAPTIONS_JSON.unlink()
        print("🗑️ Removed old captions.json")
    if PROC_DIR.exists():
        shutil.rmtree(PROC_DIR)
        print("🗑️ Removed old processed dataset")



def preprocess_dataset(images_dir: Path, captions_json: Path, processed_dir: Path):
    processed_dir = Path(processed_dir)

    with open(captions_json, "r") as f:
        captions_data = json.load(f)

    if "image" in captions_data:
        print("⚠️ Removing junk key 'image' from captions.json")
        del captions_data["image"]

    available_images = {p.name for p in Path(images_dir).glob("*.jpg")}
    fixed_captions = {img: caps for img, caps in captions_data.items() if img in available_images}


    dropped = len(captions_data) - len(fixed_captions)
    print(f"⚠️ Dropped {dropped} entries (no matching image file found).")

    all_images = list(fixed_captions.keys())
    print(len)
    if len(all_images) == 0:
        raise ValueError("❌ No valid images found. Check filenames in captions.json and data/images/")

    train_imgs, temp_imgs = train_test_split(all_images, test_size=0.2, random_state=42)
    val_imgs, test_imgs = train_test_split(temp_imgs, test_size=0.5, random_state=42)

    splits = {
        "train": {img: fixed_captions[img] for img in train_imgs},
        "val": {img: fixed_captions[img] for img in val_imgs},
        "test": {img: fixed_captions[img] for img in test_imgs},
    }

    processed_dir.mkdir(parents=True, exist_ok=True)
    for split, data in splits.items():
        with open(processed_dir / f"{split}.json", "w") as f:
            json.dump(data, f)

    print(f"✅ Preprocessing complete: {len(train_imgs)} train, {len(val_imgs)} val, {len(test_imgs)} test")

# ----------------------------
# 3️⃣ Device selection
# ----------------------------
def get_device():
    if torch.cuda.is_available():
        print("Using CUDA GPU")
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        print("Using Apple M1/M2 GPU (MPS)")
        return torch.device("mps")
    else:
        print("Using CPU")
        return torch.device("cpu")

# ----------------------------
# 4️⃣ Run scripts helper
# ----------------------------
def run_script(script_path: Path, python_exe="python3", env=None):
    print(f"\n🚀 Running {script_path}...\n")
    result = subprocess.run([python_exe, str(script_path)], env=env)
    if result.returncode != 0:
        raise RuntimeError(f"❌ Error running {script_path}. Exit code: {result.returncode}")

# ----------------------------
# 5️⃣ Main pipeline
# ----------------------------
if __name__ == "__main__":
    print("🧹 Cleaning project...")
    remove_pycache_and_pyc(BASE_DIR)
    remove_experiment_data(BASE_DIR / "experiments")
    remove_temp_files(BASE_DIR)
    remove_dataset_artifacts()   # 👈 added
    print("✅ Cleanup complete!")

    # Device
    device = get_device()
    env = os.environ.copy()
    env["DEVICE"] = str(device)

    # Step 0: Always regenerate captions.json from captions.txt
    if CAPTIONS_TXT.exists():
        print("⚡ Generating fresh captions.json from captions.txt...")
        prepare_captions(CAPTIONS_TXT, CAPTIONS_JSON)
    else:
        raise FileNotFoundError("❌ captions.txt not found in data/")
    
    # # Step 1: Preprocess captions/images
    # preprocess_dataset(IMAGES_DIR, CAPTIONS_JSON, PROC_DIR)

    # # Step 2: Run training
    # train_script = BASE_DIR / "train.py"
    # if not train_script.exists():
    #     raise FileNotFoundError(f"Training script not found at {train_script}")
    # run_script(train_script, python_exe="python3", env=env)

    print("\n✅ Pipeline completed successfully!")
