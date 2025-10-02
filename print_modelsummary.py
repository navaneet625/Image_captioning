# print_modelsummary.py
import torch
from torchinfo import summary
import yaml
from src.encoder import EncoderResNet50
from src.decoder import DecoderWithAttention
from src.dataset import CaptionDataset

def load_config(path="configs.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def print_model_summary(config_path="configs.yaml", batch_size=4, device="cpu"):
    cfg = load_config(config_path)
    device = torch.device(device)

    # ----- Encoder -----
    encoder = EncoderResNet50(pretrained=False).to(device)
    dummy_images = torch.randn(batch_size, 3, 224, 224).to(device)
    encoder_out = encoder(dummy_images)
    print("✅ Encoder summary:")
    summary(encoder, input_data=dummy_images, verbose=2)

    # ----- Decoder -----
    vocab_size = 5000 
    if "path" in cfg:
        ds = CaptionDataset(
            images_dir=cfg["data"]["images_dir"],
            captions_json=cfg["data"]["captions_json"],
            min_freq=cfg["data"].get("min_freq", 1),
            max_len=cfg["model"].get("max_len", 20)
        )
        vocab_size = len(ds.vocab)

    decoder = DecoderWithAttention(
        attention_dim=cfg["model"]["attention_dim"],
        embed_dim=cfg["model"]["embed_dim"],
        decoder_dim=cfg["model"]["decoder_dim"],
        vocab_size=vocab_size,
        encoder_dim=encoder.out_dim,
        dropout=cfg["model"].get("dropout", 0.5)
    ).to(device)

    dummy_captions = torch.randint(0, vocab_size, (batch_size, cfg["model"].get("max_len", 20))).to(device)
    print("\n✅ Decoder summary:")
    summary(decoder, input_data=(encoder_out, dummy_captions), verbose=2)

if __name__ == "__main__":
    print_model_summary()
