import os
import yaml
from pathlib import Path
import torch
from torch.utils.data import DataLoader

from src.dataset import CaptionDataset
from src.encoder import EncoderCNN
from src.decoder import DecoderWithAttention

# -----------------------------
# Paths
# -----------------------------
BASE_DIR = Path("/content/drive/MyDrive/image_caption")
DATA_DIR = BASE_DIR / "data" / "processed"
EXPERIMENTS_DIR = BASE_DIR / "experiments"
CHECKPOINT_DIR = EXPERIMENTS_DIR / "exp1" 
RESULTS_DIR = EXPERIMENTS_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------
# Load config
# -----------------------------
cfg_path = BASE_DIR / "src" / "configs.yaml"
with open(cfg_path, "r") as f:
    cfg = yaml.safe_load(f)

# Update dataset paths for testing
cfg["data"]["images_dir"] = str(BASE_DIR / "data" / "Images")
cfg["data"]["captions_json"] = str(DATA_DIR / "test.json")  # or val.json

# -----------------------------
# Device
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() and cfg.get("use_cuda", True) else "cpu")
print(f"Using device: {device}")

# -----------------------------
# Dataset & Dataloader
# -----------------------------
ds = CaptionDataset(
    images_dir=cfg["data"]["images_dir"],
    captions_json=cfg["data"]["captions_json"],
    min_freq=cfg["data"].get("min_freq", 1),
    max_len=cfg["model"].get("max_len", 20)
)
vocab = ds.vocab
pad_idx = vocab.pad_idx

loader = DataLoader(ds, batch_size=1, shuffle=False, collate_fn=CaptionDataset.collate_fn, num_workers=0)

# -----------------------------
# Models
# -----------------------------
encoder = EncoderCNN(pretrained=True, trainable=False).to(device)
decoder = DecoderWithAttention(
    attention_dim=int(cfg["model"]["attention_dim"]),
    embed_dim=int(cfg["model"]["embed_dim"]),
    decoder_dim=int(cfg["model"]["decoder_dim"]),
    vocab_size=len(vocab),
    encoder_dim=encoder.out_dim,
    dropout=float(cfg["model"].get("dropout", 0.5))
).to(device)

# -----------------------------
# Load checkpoint
# -----------------------------
checkpoint_epoch = 20  # choose which checkpoint to load
ckpt_path = CHECKPOINT_DIR / f"ckpt_epoch{checkpoint_epoch}.pth"
ckpt = torch.load(ckpt_path, map_location=device)

encoder.load_state_dict(ckpt["encoder_state"])
decoder.load_state_dict(ckpt["decoder_state"])
decoder.eval()
encoder.eval()
print(f"✅ Loaded checkpoint from epoch {checkpoint_epoch}")

# -----------------------------
# Generate captions
# -----------------------------
results = {}
with torch.no_grad():
    for images, caps, lengths, img_names in loader:
        images = images.to(device)
        encoder_out = encoder(images)
        outputs, _ = decoder.sample(encoder_out)  # assumes decoder.sample() generates predicted captions

        # Convert indices to words
        captions_str = []
        for output in outputs:
            words = [vocab.idx2word[idx.item()] for idx in output if idx.item() != pad_idx]
            captions_str.append(" ".join(words))

        results[img_names[0]] = captions_str

# -----------------------------
# Save results
# -----------------------------
import json
results_path = RESULTS_DIR / f"predicted_captions_epoch{checkpoint_epoch}.json"
with open(results_path, "w") as f:
    json.dump(results, f, indent=4)

print(f"✅ Test captions saved at {results_path}")
