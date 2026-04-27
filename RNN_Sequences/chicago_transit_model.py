from torch import nn
from initializors import kaiming_normal_initialize

INPUT_SIZE = 4
OUTPUT_SIZE = 1

class ChicagoRidershipModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super().__init__()
        self.hidden_size = hidden_size
        self.memory_cell = nn.LSTM(input_size, hidden_size, batch_first=True, num_layers=num_layers)
        self.output = nn.Sequential(nn.Linear(hidden_size, hidden_size * 2), nn.ReLU(),
                                    nn.Linear(hidden_size * 2, hidden_size * 2), nn.ReLU(),
                                    nn.Linear(hidden_size * 2, output_size))
        kaiming_normal_initialize(self.output)

    def forward(self, X):
        outputs, (h, c) = self.memory_cell(X)
        return self.output(outputs[:, -1])
