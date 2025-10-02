import os
import yaml
import json
import torch
import sys
from torch.utils.data import DataLoader
from pathlib import Path

PROJECT_ROOT = Path("/kaggle/working/imgc")
sys.path.append(str(PROJECT_ROOT))

from src.dataset import CaptionDataset, Vocabulary
from src.encoder import EncoderResNet50
from src.decoder import DecoderWithAttention

# update path according to dir structure ......
BASE_DIR = Path("/kaggle/working/imgc") 

DATA_DIR = BASE_DIR / "data" / "processed"
EXP_DIR = BASE_DIR / "experiments" / "checkpoints"
CHECKPOINT_PATH = EXP_DIR / "epoch_40.pth" 
RESULTS_DIR = BASE_DIR / "experiments" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


ckpt = torch.load(CHECKPOINT_PATH, map_location=device)
print(f"✅ Loaded checkpoint from {CHECKPOINT_PATH.name}")

# Restore vocabulary
vocab_word2idx = ckpt["vocab"]
vocab = Vocabulary()
vocab.word2idx = vocab_word2idx
vocab.idx2word = {idx: word for word, idx in vocab_word2idx.items()}


CONFIG_PATH = BASE_DIR / "configs.yaml"
with open(CONFIG_PATH, "r") as f:
    cfg = yaml.safe_load(f)


cfg["data"]["images_dir"] = str(BASE_DIR / "data" / "Images")
cfg["data"]["captions_json"] = str(DATA_DIR / "test.json") 

ds = CaptionDataset(
    images_dir=cfg["data"]["images_dir"],
    captions_json=cfg["data"]["captions_json"],
    vocab=vocab,
    max_len=cfg["model"].get("max_len", 20)
)

loader = DataLoader(
    ds,
    batch_size=1, 
    shuffle=False,
    collate_fn=CaptionDataset.collate_fn,
    num_workers=0
)

encoder = EncoderResNet50(pretrained=False, trainable=False).to(device)
decoder = DecoderWithAttention(
    attention_dim=int(cfg["model"]["attention_dim"]),
    embed_dim=int(cfg["model"]["embed_dim"]),
    decoder_dim=int(cfg["model"]["decoder_dim"]),
    vocab_size=len(vocab),
    encoder_dim=encoder.out_dim,
    dropout=float(cfg["model"].get("dropout", 0.5))
).to(device)

encoder.load_state_dict(ckpt["encoder_state"])
decoder.load_state_dict(ckpt["decoder_state"])
encoder.eval()
decoder.eval()
print("✅ Models loaded and ready.")


results = {}
with torch.no_grad():
    for i, (images, _, lengths) in enumerate(loader):
        images = images.to(device)
        encoder_out = encoder(images)  # (1, L, encoder_dim)
        caption = decoder.generate(
            encoder_out,
            vocab=vocab,
            max_len=cfg["model"].get("max_len", 20),
            device=device
        )
        img_name = ds.keys[i]
        results[img_name] = caption
        print(f"{img_name}: {caption}")


with open(RESULTS_DIR / "generated_captions.json", "w") as f:
    json.dump(results, f, indent=2)

print(f"✅ Captions saved at {RESULTS_DIR / 'generated_captions.json'}")
