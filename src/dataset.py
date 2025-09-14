import os
import json
from collections import Counter

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

PAD = "<pad>"
START = "<start>"
END = "<end>"
UNK = "<unk>"

class Vocabulary:
    def __init__(self, min_freq=1):
        self.min_freq = min_freq
        self.word2idx = {}
        self.idx2word = {}
        self.freqs = Counter()
        # reserve special tokens
        self.add_word(PAD)
        self.add_word(START)
        self.add_word(END)
        self.add_word(UNK)

    def add_word(self, w):
        if w not in self.word2idx:
            idx = len(self.word2idx)
            self.word2idx[w] = idx
            self.idx2word[idx] = w
        return self.word2idx[w]

    def build_from_captions(self, captions_dict):
        # captions_dict: {filename: [caption1, caption2, ...], ...}
        for caps in captions_dict.values():
            for c in caps:
                for tok in self._tokenize(c):
                    self.freqs[tok] += 1
        for tok, freq in self.freqs.items():
            if freq >= self.min_freq:
                self.add_word(tok)

    def _tokenize(self, text):
        return text.lower().strip().split()

    def encode_caption(self, text, max_len=None):
        toks = [START] + self._tokenize(text) + [END]
        idxs = [self.word2idx.get(t, self.word2idx[UNK]) for t in toks]
        if max_len is not None:
            if len(idxs) > max_len:
                idxs = idxs[:max_len]
        return idxs

    def decode_indices(self, indices):
        words = []
        for i in indices:
            w = self.idx2word.get(i, UNK)
            if w in (START, END, PAD):
                continue
            words.append(w)
        return " ".join(words)

    @property
    def pad_idx(self):
        return self.word2idx[PAD]

    @property
    def start_idx(self):
        return self.word2idx[START]

    @property
    def end_idx(self):
        return self.word2idx[END]

    @property
    def unk_idx(self):
        return self.word2idx[UNK]

    def __len__(self):
        return len(self.word2idx)


class CaptionDataset(Dataset):
    """
    Expect project layout:
      data/
        images/    (img files)
        captions.json  -> {"img1.jpg": ["a cat on a mat", ...], ...}
    This dataset returns:
      image_tensor (PIL transforms applied), caption_idx_list (first caption)
    """

    def __init__(self, images_dir, captions_json, vocab=None, min_freq=1, transform=None, max_len=20):
        with open(captions_json, "r", encoding="utf-8") as f:
            self.captions = json.load(f)
        self.images_dir = images_dir
        self.keys = sorted(list(self.captions.keys()))
        self.max_len = max_len

        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transform

        if vocab is None:
            self.vocab = Vocabulary(min_freq)
            self.vocab.build_from_captions(self.captions)
        else:
            self.vocab = vocab

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        img_name = self.keys[idx]
        img_path = os.path.join(self.images_dir, img_name)
        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)
        # take the first caption available (you can sample others later)
        caption = self.captions[img_name][0]
        caption_idx = self.vocab.encode_caption(caption, max_len=self.max_len)
        return img, torch.tensor(caption_idx, dtype=torch.long)

    @staticmethod
    def collate_fn(batch):
        """
        batch: list of tuples (image_tensor, caption_tensor(var len))
        Returns padded images tensor (B,C,H,W), captions padded (B,T), lengths
        """
        images, captions = zip(*batch)
        images = torch.stack(images, dim=0)
        lengths = [len(c) for c in captions]
        max_len = max(lengths)
        padded = torch.full((len(captions), max_len), fill_value=0, dtype=torch.long)  # 0 should be pad idx in vocab
        for i, c in enumerate(captions):
            padded[i, :len(c)] = c
        return images, padded, torch.tensor(lengths, dtype=torch.long)
