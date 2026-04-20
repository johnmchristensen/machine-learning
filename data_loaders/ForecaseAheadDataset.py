from data_loaders.TimeSeriesDataset import TimeSeriesDataset


class ForecastAheadDataset(TimeSeriesDataset):
    def __init__(self, series, window_length, forecast_length):
        super().__init__(series, window_length)
        self.forecast_length = forecast_length

    def __len__(self):
        return len(self.series) - self.window_length - self.forecast_length + 1

    def __getitem__(self, idx):
        end = idx + self.window_length  # 1st index after window
        window = self.series[idx : end]
        target = self.series[end : end + self.forecast_length, 0]  # 0 = rail ridership
        return window, target
