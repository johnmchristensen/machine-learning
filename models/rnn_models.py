import torch
from torch import nn


class SimpleRnnModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super().__init__()
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True, num_layers=num_layers)
        self.output = nn.Linear(hidden_size, output_size)

    def forward(self, X):
        outputs, last_state = self.rnn(X)
        return self.output(outputs[:, -1])


class Seq2SeqRnnModel(SimpleRnnModel):
    def forward(self, X):
        outputs, last_state = self.rnn(X)
        return self.output(outputs)

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=2):
        super().__init__()
        self.hidden_size = hidden_size
        self.memory_cell = nn.LSTM(input_size, hidden_size, batch_first=True, num_layers=num_layers)
        self.output = nn.Linear(hidden_size, output_size)

    def forward(self, X):
        outputs, (h, c) = self.memory_cell(X)
        return self.output(outputs[:, -1])
