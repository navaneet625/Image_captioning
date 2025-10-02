import torch
import torch.nn as nn
import torch.nn.functional as F
from src.attention import Attention


class DecoderWithAttention(nn.Module):
    """
    Decoder with attention (Show, Attend and Tell).
    - embedding -> attention -> LSTMCell -> deep output layer -> vocab distribution
    """

    def __init__(self, attention_dim, embed_dim, decoder_dim, vocab_size,
                 encoder_dim=2048, dropout=0.5):
        super().__init__()

        self.encoder_dim = encoder_dim
        self.attention_dim = attention_dim
        self.embed_dim = embed_dim
        self.decoder_dim = decoder_dim
        self.vocab_size = vocab_size

        # Attention network
        self.attention = Attention(encoder_dim, decoder_dim, attention_dim)

        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embed_dim)

        # Decoder LSTM
        self.decode_step = nn.LSTMCell(embed_dim + encoder_dim, decoder_dim, bias=True)

        # Initial hidden/cell from mean of encoder features
        self.init_h = nn.Linear(encoder_dim, decoder_dim)
        self.init_c = nn.Linear(encoder_dim, decoder_dim)

        # Gate scalar β_t
        self.f_beta = nn.Linear(decoder_dim, encoder_dim)
        self.sigmoid = nn.Sigmoid()

        # Deep output layer (Eq.7: combines h_t, context, embedding)
        self.fc_h = nn.Linear(decoder_dim, embed_dim)
        self.fc_z = nn.Linear(encoder_dim, embed_dim)
        self.fc_e = nn.Linear(embed_dim, embed_dim)
        self.fc_out = nn.Linear(embed_dim, vocab_size)

        self.dropout = nn.Dropout(p=dropout)

    def init_hidden_state(self, encoder_out):
        """
        Initialize LSTM hidden & cell states using mean of encoder output.
        """
        mean_enc = encoder_out.mean(dim=1)  # (B, encoder_dim)
        h = self.init_h(mean_enc)           # (B, decoder_dim)
        c = self.init_c(mean_enc)           # (B, decoder_dim)
        return h, c

    def forward(self, encoder_out, captions):
        """
        Forward pass (teacher forcing).
        encoder_out: (B, L, encoder_dim)
        captions: (B, T) token indices
        Returns:
          outputs: (B, T-1, vocab_size)
          alphas:  (B, T-1, L)
        """
        batch_size = encoder_out.size(0)
        device = encoder_out.device
        T = captions.size(1)

        # Embed all captions
        embeddings = self.embedding(captions)  # (B, T, embed_dim)

        # Init LSTM hidden/cell
        h, c = self.init_hidden_state(encoder_out)

        # Containers
        outputs = torch.zeros(batch_size, T - 1, self.vocab_size, device=device)
        alphas = torch.zeros(batch_size, T - 1, encoder_out.size(1), device=device)

        for t in range(T - 1):
            emb_t = embeddings[:, t, :]  # (B, embed_dim)

            # Attention context
            context, alpha = self.attention(encoder_out, h)  # (B, encoder_dim), (B, L)
            gate = self.sigmoid(self.f_beta(h))              # (B, encoder_dim)
            context = gate * context

            # LSTM step
            lstm_input = torch.cat([emb_t, context], dim=1)  # (B, embed_dim + encoder_dim)
            h, c = self.decode_step(lstm_input, (h, c))      # (B, decoder_dim)

            # Deep output layer: combine h, context, emb
            deep_out = self.fc_h(h) + self.fc_z(context) + self.fc_e(emb_t)
            preds = self.fc_out(self.dropout(torch.tanh(deep_out)))

            outputs[:, t, :] = preds
            alphas[:, t, :] = alpha

        return outputs, alphas

    def generate(self, encoder_out, vocab, max_len=20, device="cpu"):
        """
        Greedy decoding for inference.
        encoder_out: (1, L, encoder_dim) for single image
        """
        self.eval()
        with torch.no_grad():
            h, c = self.init_hidden_state(encoder_out)
            word = torch.tensor([vocab.start_idx], device=device).long()  # <start>
            seq = []

            for _ in range(max_len):
                emb = self.embedding(word).squeeze(1)  # (1, embed_dim)
                context, alpha = self.attention(encoder_out, h)
                gate = self.sigmoid(self.f_beta(h))
                context = gate * context

                lstm_input = torch.cat([emb, context], dim=1)
                h, c = self.decode_step(lstm_input, (h, c))

                # Deep output
                deep_out = self.fc_h(h) + self.fc_z(context) + self.fc_e(emb)
                preds = self.fc_out(torch.tanh(deep_out))  # (1, vocab_size)

                _, next_word = preds.max(dim=1)
                next_idx = next_word.item()
                if next_idx == vocab.end_idx:
                    break
                seq.append(next_idx)
                word = next_word.unsqueeze(0)

            return vocab.decode_indices(seq)
