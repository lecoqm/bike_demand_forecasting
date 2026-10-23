"""Load the original UCI hourly data onto a complete hourly calendar."""

import hashlib
from io import BytesIO
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile

import pandas as pd

DATA_URL = "https://archive.ics.uci.edu/static/public/275/bike+sharing+dataset.zip"
# Original UCI bytes and the identical CSV with Git-normalized LF line endings.
ALLOWED_SHA256 = {
    "e03de4ee4ef4dc376ac6e04bf829673c6269e8eba5c60fa121640fa2f829504f",
    "b03a2d02e8c10f435c43c7f0b358b7e34a003afea53dbc37f0183f2763295133",
}


def load_hourly_data(path: Path) -> pd.DataFrame:
    """Download if needed, verify the CSV, and retain gaps as missing counts.

    Timestamps use the original dataset's timezone-naive local clock labels.
    """
    if path.exists():
        csv_bytes = path.read_bytes()
    else:
        with urlopen(DATA_URL, timeout=30) as response:
            archive_bytes = response.read()
        with ZipFile(BytesIO(archive_bytes)) as archive:
            csv_bytes = archive.read("hour.csv")

    if hashlib.sha256(csv_bytes).hexdigest() not in ALLOWED_SHA256:
        raise ValueError(
            f"{path} does not match the original UCI CSV. "
            "Use the original file or remove the local copy to download it again."
        )
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(csv_bytes)

    raw = pd.read_csv(BytesIO(csv_bytes), usecols=["dteday", "hr", "cnt"])
    timestamps = pd.to_datetime(raw["dteday"]) + pd.to_timedelta(raw["hr"], unit="h")
    if not timestamps.is_unique:
        raise ValueError("Duplicate hourly timestamps in the dataset.")

    hourly = pd.DataFrame({"cnt": raw["cnt"].to_numpy()}, index=timestamps)
    hourly = hourly.sort_index()
    calendar = pd.date_range(
        hourly.index.min(), hourly.index.max(), freq="h", name="timestamp"
    )
    return hourly.reindex(calendar)
