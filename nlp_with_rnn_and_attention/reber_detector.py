from reber_dataset import ReberDataset
from torch import nn

class EmbeddedReberDetector(nn.Module):
    def __init__(self, embedding_size = 10, hidden_size = 128, layer_count = 2, dropout = 0.1):
        super().__init__()
        vocab_size = len(ReberDataset.TOKENS)
        self._embed = nn.Embedding(vocab_size, embedding_dim=embedding_size, padding_idx=ReberDataset.PAD_ID)
        self._gru = nn.GRU(input_size=embedding_size, hidden_size=hidden_size, num_layers=layer_count, batch_first=True, dropout=dropout)
        self._output = nn.Linear(hidden_size, 1)

    def forward(self, x):
        lengths = (x != ReberDataset.PAD_ID).sum(dim=1) # [B] real (non-pad) length
        embeddings = self._embed(x) # [B, T, E]
        outputs, _states = self._gru(embeddings) # [B, T, H]
        idx = (lengths - 1).view(-1, 1, 1).expand(-1, 1, outputs.size(2)) # [B, 1, H]
        last = outputs.gather(1, idx).squeeze(1) # [B, H] last NON-pad step per seq
        return self._output(last) # [B, 1]