from data_loaders.TimeSeriesDataset import TimeSeriesDataset


class MulvarTimeSeriesDataset(TimeSeriesDataset):
    def __getitem__(self, idx):
        window, target = super().__getitem__(idx)
        return window, target[:1]
