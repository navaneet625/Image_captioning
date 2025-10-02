import torch
import yaml
import os, shutil, subprocess, argparse
from pathlib import Path
from src.data_preprocess import prepare_captions, preprocess_dataset

def load_config(path: Path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def get_project_root(config_path: Path):
    return config_path.resolve().parent

def remove_pycache_and_pyc(root="."):
    for dirpath, dirnames, filenames in os.walk(root):
        if "__pycache__" in dirnames:
            shutil.rmtree(os.path.join(dirpath, "__pycache__"))
        for file in filenames:
            if file.endswith(".pyc"):
                os.remove(os.path.join(dirpath, file))

def remove_experiment_data(root: Path):
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)


def remove_temp_files(root="."):
    patterns = ["*.tmp", "*.log"]
    for pattern in patterns:
        for file_path in Path(root).rglob(pattern):
            file_path.unlink()


def remove_dataset_artifacts(captions_json: Path, processed_dir: Path):
    if captions_json.exists():
        captions_json.unlink()
        print("🗑️ Removed old captions.json")
    if processed_dir.exists():
        shutil.rmtree(processed_dir)
        print("🗑️ Removed old processed dataset")

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


def run_script(script_path: Path, python_exe="python3", env=None):
    print(f"\n🚀 Running {script_path}...\n")
    result = subprocess.run([python_exe, str(script_path)], env=env)
    if result.returncode != 0:
        raise RuntimeError(f"❌ Error running {script_path}. Exit code: {result.returncode}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
    "--config",
    default=str(Path(__file__).resolve().parent / "configs.yaml"),
    help="Path to config file"
)

    args = parser.parse_args()

    cfg = load_config(Path(args.config))
    PROJECT_ROOT = get_project_root(Path(args.config))

    data_dir = (PROJECT_ROOT / cfg["paths"]["data_dir"]).resolve()
    images_dir = (PROJECT_ROOT / cfg["paths"]["images_dir"]).resolve()
    captions_json = (PROJECT_ROOT / cfg["paths"]["captions_json"]).resolve()
    captions_txt = (data_dir / "captions.txt").resolve()
    processed_dir = (data_dir / "processed").resolve() 
    experiments_dir = (PROJECT_ROOT / cfg["paths"]["experiments_dir"]).resolve()
    checkpoints_dir = (PROJECT_ROOT / cfg["paths"]["checkpoints_dir"]).resolve()
    results_dir = (PROJECT_ROOT / cfg["paths"]["results_dir"]).resolve()

    print("🧹 Cleaning project...")
    remove_pycache_and_pyc(PROJECT_ROOT)
    remove_experiment_data(experiments_dir)
    remove_temp_files(PROJECT_ROOT)
    remove_dataset_artifacts(captions_json, processed_dir)
    print("✅ Cleanup complete!")

    # Device
    device = get_device()
    env = os.environ.copy()
    env["DEVICE"] = str(device)

    # Step 0: Regenerate captions.json
    if captions_txt.exists():
        print("⚡ Generating fresh captions.json from captions.txt...")
        prepare_captions(captions_txt, captions_json)
    else:
        raise FileNotFoundError("❌ captions.txt not found in data/")

    # Step 1: Preprocess dataset (splits train/val/test)
    preprocess_dataset(images_dir, captions_json, processed_dir)

    # Step 2: Run training
    train_script = PROJECT_ROOT / "train.py"
    if not train_script.exists():
        raise FileNotFoundError(f"Training script not found at {train_script}")
    run_script(train_script, python_exe="python3", env=env)
    print("\n✅ Pipeline completed successfully!")

