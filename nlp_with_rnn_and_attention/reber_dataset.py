from embedded_reber_grammar import EmbeddedReberGrammar
import torch
from torch.utils.data import Dataset
import random

class ReberDataset(Dataset):
    TOKENS = ("<PAD>", "B", "T", "P", "S", "X", "V", "E")
    TOKEN_TO_ID = {token: idx for idx, token in enumerate(TOKENS)}
    ID_TO_TOKEN = {idx: token for token, idx in TOKEN_TO_ID.items()}
    PAD_ID = TOKEN_TO_ID["<PAD>"]
    NON_PAD_TOKENS = tuple(token for token in TOKENS if token != "<PAD>")

    def __init__(self, length, pad_value = "<PAD>"):
        self._pad_value = pad_value
        self._token_to_id = dict(self.TOKEN_TO_ID)
        if self._pad_value not in self._token_to_id:
            self._token_to_id[self._pad_value] = len(self._token_to_id)
        self._id_to_token = {idx: token for token, idx in self._token_to_id.items()}
        self._pad_id = self._token_to_id[self._pad_value]
        grammar = EmbeddedReberGrammar()
        self._valid = [','.join(grammar.produce()) for x in range(int(length / 2))]

        self._invalid = []

        self._max_length = max((len(s.split(",")) for s in self._valid), default=2)
        for i in range(int(length / 2)):
            source_tokens = self._valid[i % len(self._valid)].split(",") if self._valid else ["B", "E"]
            self._invalid.append(",".join(self._create_hard_negative(source_tokens)))

    def __len__(self):
        return len(self._valid) + len(self._invalid)

    def __getitem__(self, idx):
        if idx < len(self._valid):
            tokens = self.split_and_fix_lengths(self._valid[idx])
            encoded = [self._token_to_id[token] for token in tokens]
            return torch.tensor(encoded, dtype=torch.long), torch.tensor([1.0], dtype=torch.float32)

        tokens = self.split_and_fix_lengths(self._invalid[idx - len(self._valid)])
        encoded = [self._token_to_id[token] for token in tokens]
        return torch.tensor(encoded, dtype=torch.long), torch.tensor([0.0], dtype=torch.float32)

    def split_and_fix_lengths(self, s: str) -> list[str]:
        parts = s.split(",")
        if len(parts) < self._max_length:
            parts += [self._pad_value] * (self._max_length - len(parts))
        return parts

    def _create_hard_negative(self, tokens: list[str]) -> list[str]:
        mutated = list(tokens)
        mutate_idx = random.randint(0, len(mutated) - 1)
        current = mutated[mutate_idx]
        candidates = [token for token in self.NON_PAD_TOKENS if token != current]
        replacement = candidates[random.randint(0, len(candidates) - 1)]
        mutated[mutate_idx] = replacement
        return mutated

