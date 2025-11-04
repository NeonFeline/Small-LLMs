import os
import torch
from torch.utils.data import Dataset
import tiktoken
from tiktoken import Encoding

class AsciiTokenizer():
    def __init__(self, vocab_size=128):
        self.vocab_size = vocab_size  # ASCII range
        self.pad_token = 0  # Padding token index
        self.unk_token = 1  # Unknown token index

    def encode(self, text, seq_length=None):
        if seq_length is None:
            seq_length = len(text)
        tokens = [ord(c) if ord(c) < self.vocab_size else self.unk_token for c in text]
        if len(tokens) < seq_length:
            tokens += [self.pad_token] * (seq_length - len(tokens))  # Pad
        else:
            tokens = tokens[:seq_length]  # Truncate
        return tokens

    def decode(self, tokens):
        chars = [chr(t) if t < self.vocab_size else '?' for t in tokens if t != self.pad_token]
        return ''.join(chars)


class TextDataset(Dataset):
    def __init__(self, dataset_name="wikipedia", tokeknizer=None, seq_length=64):
        self.tokenizer = tokeknizer if tokeknizer else AsciiTokenizer()
        self.seq_length = seq_length
        if dataset_name == "wikipedia":
            from datasets import load_dataset
            self.dataset = load_dataset("wikimedia/wikipedia", "20231101.en")['train']
        else:
            raise ValueError(f"Dataset {dataset_name} not supported.")

    def __len__(self):
        return len(self.dataset)

    def get_pad_token(self):
        if isinstance(self.tokenizer, AsciiTokenizer):
            return self.tokenizer.pad_token
        elif isinstance(self.tokenizer, Encoding):
            return self.tokenizer.special_tokens["<|pad|>"]
        else:
            raise ValueError("Tokenizer must be either AsciiTokenizer or tiktoken.Encoding")

    def __getitem__(self, idx):
        # Check if using ascii tokenizer
        if isinstance(self.tokenizer, AsciiTokenizer):
            # select only seq_length + 1 tokens for input-target pair

            text = self.dataset[idx]
            # random int between 0 and len(text) - seq_length - 1
            if len(text['text']) < self.seq_length + 1:
                start_idx = 0
            else:
                upper_bound = len(text['text']) - self.seq_length - 1
                if upper_bound <= 0:
                    start_idx = 0
                else:
                    start_idx = torch.randint(0, upper_bound, (1,)).item()
            text['text'] = text['text'][start_idx:start_idx + self.seq_length + 1]

            code = self.tokenizer.encode(text['text'], self.seq_length)
            tokens = torch.tensor(code, dtype=torch.long)

            # pad if needed
            if len(tokens) < self.seq_length + 1:
                padding = torch.full((self.seq_length + 1 - len(tokens),), self.tokenizer.pad_token, dtype=torch.long)
                tokens = torch.cat([tokens, padding], dim=0)
        elif isinstance(self.tokenizer, Encoding):
            text = self.dataset[idx]
            tokens = enc.encode(text)
            tokens = tokens[:self.seq_length + 1]  # Truncate
            tokens = torch.tensor(tokens, dtype=torch.long)
            # pad if needed
            if len(tokens) < self.seq_length + 1:
                padding = torch.full((self.seq_length + 1 - len(tokens),), 0, dtype=torch.long)
                tokens = torch.cat([tokens, padding], dim=0)
        else:
            raise ValueError("Tokenizer must be either AsciiTokenizer or tiktoken.Encoding")

        return {"input_ids": tokens[:-1], "target_ids": tokens[1:]}  # Shifted for language modeling