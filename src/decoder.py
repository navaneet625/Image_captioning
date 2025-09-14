import torch
import torch.nn as nn
import torch.nn.functional as F
from src.attention import Attention

class DecoderWithAttention(nn.Module):
    """
    Decoder that uses attention over spatial encoder features.
    - embedding -> LSTMCell -> output projection to vocab
    """

    def __init__(self, attention_dim, embed_dim, decoder_dim, vocab_size, encoder_dim=2048, dropout=0.5):
        super().__init__()
        self.encoder_dim = encoder_dim
        self.attention_dim = attention_dim
        self.embed_dim = embed_dim
        self.decoder_dim = decoder_dim
        self.vocab_size = vocab_size

        self.attention = Attention(encoder_dim, decoder_dim, attention_dim)

        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.dropout = nn.Dropout(p=dropout)
        self.decode_step = nn.LSTMCell(embed_dim + encoder_dim, decoder_dim, bias=True)
        self.init_h = nn.Linear(encoder_dim, decoder_dim)
        self.init_c = nn.Linear(encoder_dim, decoder_dim)
        self.f_beta = nn.Linear(decoder_dim, encoder_dim)  # gating scalar, optional
        self.sigmoid = nn.Sigmoid()

        self.fc = nn.Linear(decoder_dim, vocab_size)

    def init_hidden_state(self, encoder_out):
        # encoder_out: (B, L, encoder_dim)
        mean_enc = encoder_out.mean(dim=1)  # (B, encoder_dim)
        h = self.init_h(mean_enc)          # (B, decoder_dim)
        c = self.init_c(mean_enc)
        return h, c

    def forward(self, encoder_out, captions):
        """
        encoder_out: (B, L, encoder_dim)
        captions: (B, T) token indices, each includes START, content, END
        Returns logits (B, T-1, vocab_size) - predictions for t=1..T-1 (target is t=1..T-1)
        """
        batch_size = encoder_out.size(0)
        vocab_size = self.vocab_size
        device = encoder_out.device

        embeddings = self.embedding(captions)  # (B, T, embed_dim)
        h, c = self.init_hidden_state(encoder_out)

        T = captions.size(1)
        outputs = torch.zeros(batch_size, T - 1, vocab_size, device=device)
        alphas = torch.zeros(batch_size, T - 1, encoder_out.size(1), device=device)

        # decode step by step (teacher forcing)
        for t in range(T - 1):
            # input embedding is of current token (t) and we predict next token (t+1)
            emb_t = embeddings[:, t, :]  # (B, embed_dim)
            context, alpha = self.attention(encoder_out, h)  # (B, encoder_dim), (B, L)
            gate = self.sigmoid(self.f_beta(h))  # (B, encoder_dim)
            context = gate * context

            lstm_input = torch.cat([emb_t, context], dim=1)  # (B, embed_dim + encoder_dim)
            h, c = self.decode_step(lstm_input, (h, c))
            preds = self.fc(self.dropout(h))  # (B, vocab_size)
            outputs[:, t, :] = preds
            alphas[:, t, :] = alpha

        return outputs, alphas

    def generate(self, encoder_out, vocab, max_len=20, device="cpu"):
        """
        Greedy decoding for inference.
        encoder_out: (1, L, encoder_dim) for a single image
        """
        self.eval()
        with torch.no_grad():
            h, c = self.init_hidden_state(encoder_out)
            word = torch.tensor([vocab.start_idx], device=device).long()
            seq = []
            for _ in range(max_len):
                emb = self.embedding(word).squeeze(0)  # (embed_dim,)
                context, alpha = self.attention(encoder_out, h)
                gate = self.sigmoid(self.f_beta(h))
                context = gate * context
                lstm_input = torch.cat([emb, context.squeeze(0)], dim=-1).unsqueeze(0)  # (1, input_dim)
                h, c = self.decode_step(lstm_input.squeeze(0), (h, c))
                preds = self.fc(h)  # (1, vocab_size) or (vocab_size,)
                _, next_word = preds.max(dim=1)
                next_idx = next_word.item()
                if next_idx == vocab.end_idx:
                    break
                seq.append(next_idx)
                word = next_word
            # convert list of idx to words
            return vocab.decode_indices(seq)
