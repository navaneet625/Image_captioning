import os
import argparse
import yaml
import time
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.dataset import CaptionDataset
from src.encoder import EncoderCNN
from src.decoder import DecoderWithAttention

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" / "processed"
EXPERIMENTS_DIR = BASE_DIR / "experiments"
CHECKPOINT_DIR = EXPERIMENTS_DIR / "checkpoints"
RESULTS_DIR = EXPERIMENTS_DIR / "results"

CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------
# Utility functions
# -----------------------------
def save_checkpoint(state, path):
    torch.save(state, path)

def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

# -----------------------------
# Main training loop
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(BASE_DIR / "configs.yaml"))
    parser.add_argument("--exp_dir", default=str(CHECKPOINT_DIR), help="Folder to save checkpoints")
    args = parser.parse_args()

    cfg = load_config(args.config)
    os.makedirs(args.exp_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() and cfg.get("use_cuda", True) else "cpu")
    print(f"Using device: {device}")

    # -----------------------------
    # Dataset
    # -----------------------------
    cfg["data"]["images_dir"] = str(BASE_DIR / "data" / "Images")
    cfg["data"]["captions_json"] = str(DATA_DIR / "train.json")

    ds = CaptionDataset(
        images_dir=cfg["data"]["images_dir"],
        captions_json=cfg["data"]["captions_json"],
        min_freq=cfg["data"].get("min_freq", 1),
        max_len=cfg["model"].get("max_len", 20)
    )
    vocab = ds.vocab
    pad_idx = vocab.pad_idx

    loader = DataLoader(ds, batch_size=int(cfg["training"]["batch_size"]),
                        shuffle=True,
                        collate_fn=CaptionDataset.collate_fn,
                        num_workers=0)

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

    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
    optimizer = torch.optim.Adam(decoder.parameters(), lr=float(cfg["training"]["lr"]))

    num_epochs = int(cfg["training"]["epochs"])
    print_every = int(cfg["training"].get("print_every", 10))
    grad_clip = float(cfg["training"].get("grad_clip", 5.0))
    global_step = 0

    # -----------------------------
    # Training loop
    # -----------------------------
    for epoch in range(num_epochs):
        decoder.train()
        epoch_loss = 0.0
        start_time = time.time()

        for i, (images, captions, lengths) in enumerate(loader):
            images, captions = images.to(device), captions.to(device)

            # forward
            encoder_out = encoder(images)
            outputs, alphas = decoder(encoder_out, captions)

            # compute loss
            targets = captions[:, 1:]
            B, Tm1, V = outputs.shape
            loss = criterion(outputs.view(B * Tm1, V), targets.contiguous().view(-1))

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(decoder.parameters(), grad_clip)
            optimizer.step()

            epoch_loss += loss.item()
            global_step += 1

            if global_step % print_every == 0:
                print(f"[Epoch {epoch+1}/{num_epochs}] Step {global_step} Loss: {loss.item():.4f}")

        elapsed = time.time() - start_time
        avg_loss = epoch_loss / len(loader)
        print(f"Epoch {epoch+1} finished in {elapsed:.1f}s — avg loss {avg_loss:.4f}")

        # -----------------------------
        # Save checkpoint
        # -----------------------------
        ckpt = {
            "epoch": epoch + 1,
            "encoder_state": encoder.state_dict(),
            "decoder_state": decoder.state_dict(),
            "vocab": vocab.word2idx,
            "cfg": cfg
        }
        ckpt_path = os.path.join(args.exp_dir, "latest.pth")
        save_checkpoint(ckpt, ckpt_path)

        if (epoch + 1) % 10 == 0 or (epoch + 1) == num_epochs:
            ckpt_path = os.path.join(args.exp_dir, f"ckpt_epoch{epoch+1}.pth")
            save_checkpoint(ckpt, ckpt_path)
            print(f"✅ Saved checkpoint at epoch {epoch+1} → {ckpt_path}")


if __name__ == "__main__":
    main()
