import argparse
import yaml
import os
import torch
from PIL import Image
from torchvision import transforms

from encoder import EncoderCNN
from decoder import DecoderWithAttention
from dataset import Vocabulary

def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def build_vocab_from_dict(word2idx):
    vocab = Vocabulary(min_freq=1)
    # override built maps
    vocab.word2idx = word2idx
    vocab.idx2word = {v:k for k,v in word2idx.items()}
    return vocab

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs.yaml")
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--max_len", type=int, default=20)
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() and cfg.get("use_cuda", True) else "cpu")

    ckpt = torch.load(args.ckpt, map_location=device)
    word2idx = ckpt["vocab"]
    vocab = build_vocab_from_dict(word2idx)

    encoder = EncoderCNN(pretrained=True, trainable=False).to(device)
    decoder = DecoderWithAttention(
        attention_dim=cfg["model"]["attention_dim"],
        embed_dim=cfg["model"]["embed_dim"],
        decoder_dim=cfg["model"]["decoder_dim"],
        vocab_size=len(vocab),
        encoder_dim=encoder.out_dim,
        dropout=cfg["model"].get("dropout", 0.5)
    ).to(device)

    encoder.load_state_dict(ckpt["encoder_state"])
    decoder.load_state_dict(ckpt["decoder_state"])

    # image preprocessing (same as dataset)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    img = Image.open(args.image).convert("RGB")
    img_t = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        enc_out = encoder(img_t)  # (1, L, D)
        caption = decoder.generate(enc_out, vocab=vocab, max_len=args.max_len, device=device)
    print("Caption:", caption)

if __name__ == "__main__":
    main()
