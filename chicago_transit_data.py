from pathlib import Path
import pandas as pd
import tarfile
import urllib.request

RIDERSHIP_URL = "https://github.com/ageron/data/raw/main/ridership.tgz"
DEFAULT_TARGET_PATH = "datasets"

def download_and_extract_ridership_data(target_path = DEFAULT_TARGET_PATH, url = RIDERSHIP_URL):
    tarball_path = Path(target_path + "/ridership.tgz")
    if not tarball_path.is_file():
        Path(target_path).mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(url, tarball_path)
        with tarfile.open(tarball_path) as housing_tarball:
            housing_tarball.extractall(path=target_path, filter="data")

def load_ridership_data(target_path = DEFAULT_TARGET_PATH):
    path = Path(target_path + "/ridership/CTA_-_Ridership_-_Daily_Boarding_Totals.csv")
    df = pd.read_csv(path, parse_dates=["service_date"])
    df.columns = ["date", "day_type", "bus", "rail", "total"]  # shorter names
    df = df.sort_values("date").set_index("date")
    df = df.drop("total", axis=1)  # no need for total, it's just bus + rail
    df = df.drop_duplicates()  # remove duplicated months (2011-10 and 2014-07)
    return df