import argparse, time, yaml
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.dataset import CaptionDataset
from src.encoder import EncoderResNet50  
from src.decoder import DecoderWithAttention

def save_checkpoint(state, path):
    torch.save(state, path)

def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs.yaml", help="Path to YAML config")
    args = parser.parse_args()

    # Load config
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = Path(__file__).resolve().parent / config_path

    cfg = load_config(config_path)
    PROJECT_ROOT = config_path.parent

    # Resolve paths
    images_dir = (PROJECT_ROOT / cfg["paths"]["images_dir"]).resolve()
    captions_json = (PROJECT_ROOT / cfg["paths"]["captions_json"]).resolve()
    experiments_dir = (PROJECT_ROOT / cfg["paths"]["experiments_dir"]).resolve()
    checkpoints_dir = (PROJECT_ROOT / cfg["paths"]["checkpoints_dir"]).resolve()
    results_dir = (PROJECT_ROOT / cfg["paths"]["results_dir"]).resolve()

    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() and cfg.get("use_cuda", True) else "cpu")
    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    print(f"Using device: {device} | GPUs available: {num_gpus}")

    ds = CaptionDataset(
        images_dir=images_dir,
        captions_json=captions_json,
        min_freq=cfg["data"].get("min_freq", 1),
        max_len=cfg["model"].get("max_len", 20)
    )
    vocab = ds.vocab
    pad_idx = vocab.pad_idx

    loader = DataLoader(
        ds,
        batch_size=int(cfg["training"]["batch_size"]),
        shuffle=True,
        collate_fn=CaptionDataset.collate_fn,
        num_workers=2
    )

    encoder = EncoderResNet50(
        pretrained=True,
        trainable=cfg["model"].get("encoder_trainable", False)
    ).to(device)

    decoder = DecoderWithAttention(
        attention_dim=int(cfg["model"]["attention_dim"]),
        embed_dim=int(cfg["model"]["embed_dim"]),
        decoder_dim=int(cfg["model"]["decoder_dim"]),
        vocab_size=len(vocab),
        encoder_dim=encoder.out_dim,
        dropout=float(cfg["model"].get("dropout", 0.5))
    ).to(device)

    # ✅ Multi-GPU support (DataParallel)
    if num_gpus > 1:
        print(f"⚡ Using {num_gpus} GPUs with DataParallel")
        encoder = nn.DataParallel(encoder)
        decoder = nn.DataParallel(decoder)


    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
    params = list(decoder.parameters()) + list(p for p in encoder.parameters() if p.requires_grad)
    optimizer = torch.optim.Adam(params, lr=float(cfg["training"]["lr"]))


    num_epochs = int(cfg["training"]["epochs"])
    print_every = int(cfg["training"].get("print_every", 50))
    grad_clip = float(cfg["training"].get("grad_clip", 5.0))
    attn_reg_lambda = float(cfg["training"].get("attn_reg_lambda", 1.0))
    global_step = 0

    for epoch in range(num_epochs):
        decoder.train()
        encoder.train()
        epoch_loss = 0.0
        start_time = time.time()

        for i, (images, captions, lengths) in enumerate(loader):
            images, captions = images.to(device), captions.to(device)

            # forward
            encoder_out = encoder(images)                      # (B, L, D)
            outputs, alphas = decoder(encoder_out, captions)   # (B, T-1, V), (B, T-1, L)

            # targets: next word
            targets = captions[:, 1:]
            B, Tm1, V = outputs.shape

            ce_loss = criterion(outputs.view(B * Tm1, V), targets.contiguous().view(-1))

            # Doubly stochastic attention regularizer (Eq.15)
            attn_reg = ((1.0 - alphas.sum(dim=1))**2).mean()
            loss = ce_loss + attn_reg_lambda * attn_reg

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(params, grad_clip)
            optimizer.step()

            epoch_loss += loss.item()
            global_step += 1

            if global_step % print_every == 0:
                print(f"[Epoch {epoch+1}/{num_epochs}] Step {global_step} "
                      f"Loss: {loss.item():.4f} (CE {ce_loss.item():.4f}, AttnReg {attn_reg.item():.4f})")

        elapsed = time.time() - start_time
        avg_loss = epoch_loss / len(loader)
        print(f"Epoch {epoch+1} | Time {elapsed:.1f}s | Avg Loss {avg_loss:.4f}")

        # Save checkpoint
        ckpt = {
            "epoch": epoch + 1,
            "encoder_state": encoder.module.state_dict() if isinstance(encoder, nn.DataParallel) else encoder.state_dict(),
            "decoder_state": decoder.module.state_dict() if isinstance(decoder, nn.DataParallel) else decoder.state_dict(),
            "vocab": vocab.word2idx,
            "cfg": cfg
        }
        save_checkpoint(ckpt, checkpoints_dir / "latest.pth")

        if (epoch + 1) % 5 == 0 or (epoch + 1) == num_epochs:
            ckpt_path = checkpoints_dir / f"epoch_{epoch+1}.pth"
            save_checkpoint(ckpt, ckpt_path)
            print(f"✅ Saved checkpoint at epoch {epoch+1} → {ckpt_path}")


if __name__ == "__main__":
    main()

