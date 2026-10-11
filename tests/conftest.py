"""Shared test fixtures.

`make_raw` builds tiny raw-format frames (all text, original headers) from one
valid base row. The base row copies the values of the dataset's first record,
so unit tests use realistic values without loading the full file.
"""

import pandas as pd
import pytest

from src.utils.config import load_schema

BASE_ROW = {
    "Season": "Winter",
    "Cell Availability (%)": "98.045",
    "MTTR (hours)": "3.0",
    "Throughput (Mbps)": "653.5314000000001",
    "Latency (ms)": "74.33599999999998",
    "Packet Loss Rate (%)": "6.4375",
    "Call Drop Rate (%)": "1.4666714999999986",
    "Handover Success Rate (%)": "99.29",
    "Alarm Count": "13",
    "Critical Alarm Count": "6.718020299999999",
    "Parameter Changes": "18",
    "Successful Configuration Changes (%)": "91.85",
    "Data Usage (GB)": "47.49",
    "User Count": "2442",
    "Signal Strength (dBm)": "-110.12",
    "Jitter (ms)": "15.67",
    "Connection Setup Success Rate (%)": "99.0",
    "Security Incidents": "8",
    "Authentication Failures": "2",
    "Temperature (°C)": "-6.75",
    "Humidity (%)": "28.79",
    "Weather": "Clouds",
    "Issue Reported": "yes",
    "City": "Schneiderville",
    "State": "MN",
    "Zip": "46530",
    "Fault Occurrence Rate (%)": "33.582417010135224",
}


@pytest.fixture(scope="session")
def schema():
    return load_schema()


@pytest.fixture
def make_raw():
    def _make(*overrides: dict) -> pd.DataFrame:
        rows = [{**BASE_ROW, **o} for o in (overrides or ({},))]
        return pd.DataFrame(rows, dtype=object)
    return _make
