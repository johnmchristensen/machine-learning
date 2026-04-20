import torch


class TimeSeriesDataset(torch.utils.data.Dataset):
    def __init__(self, series, window_length, target_data_slice = None):
        self.series = series
        self.window_length = window_length
        self.target_data_slice = target_data_slice

    def __len__(self):
        return len(self.series) - self.window_length

    def __getitem__(self, idx):
        if idx >= len(self):
            raise IndexError("dataset index out of range")
        end = idx + self.window_length  # 1st index after window
        window = self.series[idx : end]
        target = self.series[end] if self.target_data_slice is None else self.series[end, self.target_data_slice]
        return window, target
