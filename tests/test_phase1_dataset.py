"""Phase 1 tests: the raw dataset is the one we inspected, with the schema we documented.

Later phases depend on these exact column names; if upstream ever changes,
these tests fail before anything downstream silently breaks.
"""

import pandas as pd
import pytest

from src.preprocessing.download_dataset import RAW_PATH, SHA256, sha256_of

EXPECTED_COLUMNS = [
    "Season", "Cell Availability (%)", "MTTR (hours)", "Throughput (Mbps)",
    "Latency (ms)", "Packet Loss Rate (%)", "Call Drop Rate (%)",
    "Handover Success Rate (%)", "Alarm Count", "Critical Alarm Count",
    "Parameter Changes", "Successful Configuration Changes (%)", "Data Usage (GB)",
    "User Count", "Signal Strength (dBm)", "Jitter (ms)",
    "Connection Setup Success Rate (%)", "Security Incidents",
    "Authentication Failures", "Temperature (°C)", "Humidity (%)", "Weather",
    "Issue Reported", "City", "State", "Zip", "Fault Occurrence Rate (%)",
]

pytestmark = pytest.mark.skipif(
    not RAW_PATH.exists(), reason="raw dataset not downloaded (python -m src.preprocessing.download_dataset)"
)


@pytest.fixture(scope="module")
def df():
    return pd.read_csv(RAW_PATH)


def test_checksum_matches_pinned_revision():
    assert sha256_of(RAW_PATH) == SHA256


def test_shape(df):
    assert df.shape == (100_000, 27)


def test_exact_column_names_and_order(df):
    assert list(df.columns) == EXPECTED_COLUMNS


def test_no_missing_values(df):
    assert int(df.isna().sum().sum()) == 0


def test_no_original_incident_fields(df):
    # The dataset has no incident id / category / severity / text fields:
    # anything like that in later phases must be derived and labelled as such.
    for forbidden in ["incident_id", "category", "severity", "priority",
                      "description", "summary", "root cause", "recommendation"]:
        assert not any(forbidden in c.lower() for c in df.columns)


def test_categorical_domains(df):
    assert set(df["Season"]) == {"Winter", "Spring", "Summer", "Fall"}
    assert set(df["Weather"]) == {"Clear", "Clouds", "Rain", "Snow"}
    assert set(df["Issue Reported"]) == {"yes", "no"}


def test_percent_columns_within_0_100(df):
    for c in [c for c in df.columns if "(%)" in c]:
        assert df[c].between(0, 100).all(), c
