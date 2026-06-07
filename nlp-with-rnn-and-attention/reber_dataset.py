from embedded_reber_grammar import EmbeddedReberGrammar, ReberGrammar
from torch.utils.data import Dataset
import random

class ReberDataset(Dataset):
    def __init__(self, length):
        grammar = EmbeddedReberGrammar()
        self._valid = [','.join(grammar.produce()) for x in range(int(length / 2))]

        letters = ReberGrammar().get_edge_labels()
        self._invalid = []

        max_length = max(map(len, self._valid), default=2)
        print(max_length)
        for i in range(int(length / 2)):
            invalid_grammar_string = ""
            for j in range(random.randint(1, max_length)):
                invalid_grammar_string += random.choice(letters) + ','
            self._invalid.append(invalid_grammar_string)

    def __len__(self):
        return len(self._valid) + len(self._invalid)

    def __getitem__(self, idx):
        if idx <= len(self._valid):
            return self._valid[idx], True

        return self._invalid[idx - len(self._valid)], False