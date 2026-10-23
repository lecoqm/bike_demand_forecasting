"""Verify checksum handling and preserve absent clock hours."""

import hashlib

import pandas as pd
import pytest

from bike_demand import data


@pytest.fixture
def csv_content(monkeypatch):
    content = b"dteday,hr,cnt\n2012-01-01,0,10\n2012-01-01,2,30\n"
    variants = [content, content.replace(b"\n", b"\r\n")]
    monkeypatch.setattr(
        data,
        "ALLOWED_SHA256",
        {hashlib.sha256(value).hexdigest() for value in variants},
    )
    return content


@pytest.mark.parametrize("newline", [b"\n", b"\r\n"])
def test_verified_line_endings_preserve_missing_hours(tmp_path, csv_content, newline):
    path = tmp_path / "hour.csv"
    path.write_bytes(csv_content.replace(b"\n", newline))

    hourly = data.load_hourly_data(path)

    assert len(hourly) == 3
    assert hourly.loc["2012-01-01 00:00:00", "cnt"] == 10
    assert pd.isna(hourly.loc["2012-01-01 01:00:00", "cnt"])
    assert hourly.loc["2012-01-01 02:00:00", "cnt"] == 30


def test_changed_counts_are_rejected(tmp_path, csv_content):
    path = tmp_path / "hour.csv"
    path.write_bytes(csv_content.replace(b",10\n", b",11\n"))

    with pytest.raises(ValueError, match="does not match the original UCI CSV"):
        data.load_hourly_data(path)
