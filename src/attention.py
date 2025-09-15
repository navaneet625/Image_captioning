import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):
    """
    Attention module (additive style).
    Takes encoder features (B, L, D) and decoder hidden (B, D_dec) and produces
    context vector (B, D) and alpha weights (B, L).
    """

    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        super().__init__()
        self.encoder_att = nn.Linear(encoder_dim, attention_dim)  # transform encoder
        self.decoder_att = nn.Linear(decoder_dim, attention_dim)  # transform decoder
        self.full_att = nn.Linear(attention_dim, 1)               # to scalar
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, encoder_out, decoder_hidden):
        """
        encoder_out: (B, L, encoder_dim)
        decoder_hidden: (B, decoder_dim)
        returns: context (B, encoder_dim), alpha (B, L)
        """
        # (B, L, att_dim)
        enc_att = self.encoder_att(encoder_out)
        # (B, 1, att_dim) -> (B, L, att_dim) broadcast
        dec_att = self.decoder_att(decoder_hidden).unsqueeze(1)
        att = self.relu(enc_att + dec_att)            # (B, L, att_dim)
        e = self.full_att(att).squeeze(2)             # (B, L)
        alpha = self.softmax(e)                       # (B, L)
        # compute context as weighted sum
        context = (encoder_out * alpha.unsqueeze(2)).sum(dim=1)  # (B, encoder_dim)
        return context, alpha
    