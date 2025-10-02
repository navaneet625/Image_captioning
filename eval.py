import argparse
import yaml
import json
from pathlib import Path
import torch
import sys
from torch.utils.data import DataLoader
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction

PROJECT_ROOT = Path("/kaggle/working/imgc")
sys.path.append(str(PROJECT_ROOT))

from src.dataset import CaptionDataset, Vocabulary
from src.encoder import EncoderResNet50
from src.decoder import DecoderWithAttention


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def compute_bleu_and_captions(decoder, encoder, loader, vocab, device, max_len=20):
    references, hypotheses = [], []
    results = {}

    decoder.eval(); encoder.eval()
    smooth_fn = SmoothingFunction().method1 

    with torch.no_grad():
        for i, (imgs, _, _) in enumerate(loader):
            imgs = imgs.to(device)
            enc_out = encoder(imgs)

            for j in range(imgs.size(0)):
                feat = enc_out[j].unsqueeze(0)
                pred = decoder.generate(feat, vocab, max_len=max_len, device=device)
                hypotheses.append(pred.split())

                # ✅ all reference captions from dataset JSON
                img_name = loader.dataset.keys[i * loader.batch_size + j]
                ref_caps = loader.dataset.captions[img_name]
                references.append([c.split() for c in ref_caps])

                results[img_name] = pred

    # Compute BLEU scores
    bleu1 = corpus_bleu(references, hypotheses, weights=(1.0, 0, 0, 0), smoothing_function=smooth_fn)
    bleu2 = corpus_bleu(references, hypotheses, weights=(0.5, 0.5, 0, 0), smoothing_function=smooth_fn)
    bleu3 = corpus_bleu(references, hypotheses, weights=(0.33, 0.33, 0.33, 0), smoothing_function=smooth_fn)
    bleu4 = corpus_bleu(references, hypotheses, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smooth_fn)

    return (bleu1, bleu2, bleu3, bleu4), results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs.yaml", help="Path to YAML config")
    parser.add_argument("--checkpoint", default="experiments/checkpoints/latest.pth",
                        help="Path to .pth checkpoint (default: latest.pth)")
    parser.add_argument("--split", default="test", choices=["test"],
                        help="Dataset split to evaluate (now only test)")
    args = parser.parse_args([])  # for Kaggle notebooks

    config_path = PROJECT_ROOT / args.config
    cfg = load_config(config_path)

    images_dir = (PROJECT_ROOT / cfg["paths"]["images_dir"]).resolve()
    split_json = PROJECT_ROOT / "data" / "processed" / f"{args.split}.json"
    ckpt_path = PROJECT_ROOT / args.checkpoint
    results_path = PROJECT_ROOT / "experiments" / "results" / f"generated_captions_{args.split}.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    ckpt = torch.load(ckpt_path, map_location=device)
    print(f"✅ Loaded checkpoint from {ckpt_path.name}")

    vocab = Vocabulary()
    vocab.word2idx = ckpt["vocab"]
    vocab.idx2word = {idx: word for word, idx in vocab.word2idx.items()}

    ds = CaptionDataset(
        images_dir=images_dir,
        captions_json=split_json,
        vocab=vocab,
        max_len=cfg["model"].get("max_len", 20),
        mode="eval" 
    )
    loader = DataLoader(ds, batch_size=32, shuffle=False,
                        collate_fn=CaptionDataset.collate_fn)

    encoder = EncoderResNet50(pretrained=False).to(device)
    decoder = DecoderWithAttention(
        attention_dim=cfg["model"]["attention_dim"],
        embed_dim=cfg["model"]["embed_dim"],
        decoder_dim=cfg["model"]["decoder_dim"],
        vocab_size=len(vocab),
        encoder_dim=encoder.out_dim,
        dropout=cfg["model"]["dropout"]
    ).to(device)

    encoder.load_state_dict(ckpt["encoder_state"])
    decoder.load_state_dict(ckpt["decoder_state"])

    (bleu1, bleu2, bleu3, bleu4), results = compute_bleu_and_captions(
        decoder, encoder, loader, vocab, device,
        max_len=cfg["model"].get("max_len", 20)
    )

    print(f"\n✅ Final {args.split} BLEU Scores:")
    print(f"BLEU-1: {bleu1:.4f}")
    print(f"BLEU-2: {bleu2:.4f}")
    print(f"BLEU-3: {bleu3:.4f}")
    print(f"BLEU-4: {bleu4:.4f}")

    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"✅ Captions saved at {results_path}")


if __name__ == "__main__":
    main()
