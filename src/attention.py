import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):
    """
    Additive attention (Bahdanau-style) as in
    'Show, Attend and Tell' (Xu et al. 2015).
    """

    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        super().__init__()
        self.encoder_att = nn.Linear(encoder_dim, attention_dim)
        self.decoder_att = nn.Linear(decoder_dim, attention_dim)
        self.full_att = nn.Linear(attention_dim, 1)        
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, encoder_out, decoder_hidden):
        """
        encoder_out: (B, L, encoder_dim)
        decoder_hidden: (B, decoder_dim)
        returns: context (B, encoder_dim), alpha (B, L)
        """
        # (B, L, att_dim)
        enc_proj = self.encoder_att(encoder_out)     
        dec_proj = self.decoder_att(decoder_hidden).unsqueeze(1) 
        att = self.tanh(enc_proj + dec_proj)  
        e = self.full_att(att).squeeze(2)    
        alpha = self.softmax(e)                 
        context = (encoder_out * alpha.unsqueeze(2)).sum(dim=1) 
        return context, alpha
