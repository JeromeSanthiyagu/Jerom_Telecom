"""Phase 2 tests: loading, validation and cleaning.

Unit tests use tiny hand-made frames (see conftest.BASE_ROW). Full-dataset
tests prove that preprocessing never changes or invents a raw value.
"""


import numpy as np
import pandas as pd
import pytest

from src.preprocessing import cleaner
from src.preprocessing.download_dataset import RAW_PATH
from src.preprocessing.processor import assign_record_ids, clean, run_pipeline
from src.preprocessing.validator import ValidationError, validate_processed, validate_raw


def run_clean(raw, schema):
    df, log = clean(assign_record_ids(raw, schema), schema)
    return df, log


def actions(log, column=None):
    return [e["action"] for e in log.entries if column is None or e["column"] == column]


# --------------------------------------------------------------------------- #
# Column names
# --------------------------------------------------------------------------- #
def test_headers_matched_despite_case_whitespace_and_encoding(make_raw, schema):
    raw = make_raw().rename(columns={
        "Season": "  season ",
        "Latency (ms)": "LATENCY  (ms)",
        "Temperature (°C)": "Temperature (Â°C)",  # UTF-8 read as Latin-1
    })
    df, _ = run_clean(raw, schema)
    assert {"season", "latency_ms", "temperature_c"} <= set(df.columns)
    assert df.loc[0, "temperature_c"] == -6.75


def test_missing_required_column_is_an_error(make_raw, schema):
    raw = make_raw().drop(columns=["Latency (ms)"])
    assert not validate_raw(raw, schema).ok
    with pytest.raises(cleaner.SchemaError):
        run_clean(raw, schema)


def test_unknown_column_is_dropped_and_logged(make_raw, schema):
    raw = make_raw()
    raw["Comment"] = "free text"
    df, log = run_clean(raw, schema)
    assert "Comment" not in df.columns
    assert "dropped unknown columns" in actions(log)


# --------------------------------------------------------------------------- #
# Text, case, categories
# --------------------------------------------------------------------------- #
def test_whitespace_and_case_are_normalised(make_raw, schema):
    raw = make_raw({
        "Season": "  winter ", "Weather": "CLOUDS", "State": " mn",
        "City": "  New   York ", "Issue Reported": " YES ",
    })
    df, _ = run_clean(raw, schema)
    row = df.iloc[0]
    assert (row.season, row.weather, row.state, row.city) == ("Winter", "Clouds", "MN", "New York")
    assert row.issue_reported is True or row.issue_reported == True  # noqa: E712


def test_documented_aliases_are_mapped(make_raw, schema):
    df, _ = run_clean(make_raw({"Season": "Autumn", "Weather": "rainy"}), schema)
    assert (df.loc[0, "season"], df.loc[0, "weather"]) == ("Fall", "Rain")


def test_unknown_category_becomes_missing_not_guessed(make_raw, schema):
    df, log = run_clean(make_raw({"Weather": "Hail"}), schema)
    assert pd.isna(df.loc[0, "weather"])
    assert "unknown category -> NA" in actions(log, "weather")


def test_issue_reported_is_boolean(make_raw, schema):
    df, _ = run_clean(make_raw({"Issue Reported": "no"}, {"Issue Reported": "Y"}, {"Issue Reported": "maybe"}), schema)
    assert str(df["issue_reported"].dtype) == "boolean"
    assert df["issue_reported"].tolist()[:2] == [False, True]
    assert pd.isna(df.loc[2, "issue_reported"])


# --------------------------------------------------------------------------- #
# Numbers, invalid values, missing values
# --------------------------------------------------------------------------- #
def test_missing_tokens_stay_missing_and_are_not_imputed(make_raw, schema):
    df, log = run_clean(make_raw({"Latency (ms)": "N/A", "Jitter (ms)": ""}), schema)
    assert len(df) == 1
    assert pd.isna(df.loc[0, "latency_ms"]) and pd.isna(df.loc[0, "jitter_ms"])
    assert "missing tokens -> NA" in actions(log, "latency_ms")


def test_unparseable_number_becomes_missing_and_is_logged(make_raw, schema):
    df, log = run_clean(make_raw({"Throughput (Mbps)": "fast"}), schema)
    assert pd.isna(df.loc[0, "throughput_mbps"])
    assert "unparseable number -> NA" in actions(log, "throughput_mbps")


def test_numeric_text_with_spaces_is_parsed(make_raw, schema):
    df, _ = run_clean(make_raw({"Throughput (Mbps)": "  512.5 ", "User Count": "1e3"}), schema)
    assert df.loc[0, "throughput_mbps"] == 512.5
    assert df.loc[0, "user_count"] == 1000


def test_fractional_count_is_not_rounded(make_raw, schema):
    df, log = run_clean(make_raw({"Alarm Count": "3.5"}), schema)
    assert pd.isna(df.loc[0, "alarm_count"])
    assert "non-integer count -> NA" in actions(log, "alarm_count")


@pytest.mark.parametrize("raw_col,value,col", [
    ("Packet Loss Rate (%)", "150", "packet_loss_pct"),
    ("Cell Availability (%)", "-3", "cell_availability_pct"),
    ("Signal Strength (dBm)", "5", "signal_strength_dbm"),
    ("Latency (ms)", "-1", "latency_ms"),
    ("Latency (ms)", "inf", "latency_ms"),
    ("User Count", "-20", "user_count"),
    ("Temperature (°C)", "250", "temperature_c"),
])
def test_impossible_values_become_missing(make_raw, schema, raw_col, value, col):
    df, log = run_clean(make_raw({raw_col: value}), schema)
    assert pd.isna(df.loc[0, col])
    assert any("outside valid range" in a or "unparseable" in a for a in actions(log, col))


def test_valid_extremes_are_kept_unchanged(make_raw, schema):
    extremes = {
        "Call Drop Rate (%)": "0.0", "Handover Success Rate (%)": "100.0",
        "Signal Strength (dBm)": "-140", "Cell Availability (%)": "74.342",
        "Authentication Failures": "63", "Latency (ms)": "175.844",
        "Packet Loss Rate (%)": "17.0", "Alarm Count": "26",
    }
    df, log = run_clean(make_raw(extremes), schema)
    assert df.loc[0, "call_drop_pct"] == 0.0
    assert df.loc[0, "handover_success_pct"] == 100.0
    assert df.loc[0, "signal_strength_dbm"] == -140.0
    assert df.loc[0, "auth_failures"] == 63
    assert df.loc[0, "latency_ms"] == 175.844
    assert log.total("null_invalid_values") == 0


def test_zip_is_zero_padded_text(make_raw, schema):
    df, _ = run_clean(make_raw({"Zip": "503"}, {"Zip": "123456"}), schema)
    assert df.loc[0, "zip"] == "00503"
    assert pd.isna(df.loc[1, "zip"])


# --------------------------------------------------------------------------- #
# Rows: empty, duplicates, IDs, flags
# --------------------------------------------------------------------------- #
def test_row_without_any_network_evidence_is_dropped(make_raw, schema):
    empty = {c["raw"]: "" for c in schema["columns"] if c["name"] in schema["evidence_columns"]}
    df, log = run_clean(make_raw({}, empty), schema)
    assert df["record_id"].tolist() == ["ROW-000001"]
    assert log.total("drop_empty_records") == 1


def test_duplicates_dropped_and_ids_keep_raw_position(make_raw, schema):
    df, log = run_clean(make_raw({}, {"Latency (ms)": "10"}, {}), schema)
    assert df["record_id"].tolist() == ["ROW-000001", "ROW-000002"]  # ROW-000003 was the duplicate
    assert df["source_row"].tolist() == [1, 2]
    assert log.total("drop_duplicates") == 1


def test_whitespace_variants_count_as_duplicates(make_raw, schema):
    df, _ = run_clean(make_raw({}, {"City": " Schneiderville  ", "Weather": "clouds"}), schema)
    assert len(df) == 1


def test_quality_flags_annotate_without_changing_values(make_raw, schema):
    df, _ = run_clean(make_raw({"Alarm Count": "2", "Critical Alarm Count": "5.5",
                                "Cell Availability (%)": "99.0"}), schema)
    row = df.iloc[0]
    assert row.dq_critical_exceeds_total_alarms and row.dq_availability_at_cap
    assert row.alarm_count == 2 and row.critical_alarm_count == 5.5
    assert row.cell_availability_pct == 99.0


def test_processed_validation_rejects_broken_data(make_raw, schema):
    df, _ = run_clean(make_raw({}, {"Latency (ms)": "10"}), schema)
    assert validate_processed(df, schema).ok

    broken = df.copy()
    broken.loc[1, "record_id"] = "ROW-000001"
    broken.loc[0, "packet_loss_pct"] = 120.0
    broken.loc[0, "weather"] = "Fog"
    rep = validate_processed(broken, schema)
    text = " ".join(rep.errors)
    assert "record_id must be unique" in text
    assert "packet_loss_pct: values above 100" in text
    assert "weather: values outside" in text
    with pytest.raises(ValidationError):
        rep.raise_if_errors()


# --------------------------------------------------------------------------- #
# Full dataset
# --------------------------------------------------------------------------- #
full = pytest.mark.skipif(not RAW_PATH.exists(), reason="raw dataset not downloaded")


@pytest.fixture(scope="module")
def pipeline():
    from src.preprocessing.loader import load_raw
    raw = load_raw()
    df, report = run_pipeline(raw=raw, write=False)
    return raw, df, report


@full
def test_full_pipeline_keeps_every_row(pipeline):
    raw, df, report = pipeline
    assert report["rows_in"] == report["rows_out"] == 100_000
    assert report["processed_validation"]["ok"]


@full
def test_no_raw_value_is_changed_or_fabricated(pipeline, schema):
    """Every processed value equals the raw CSV text, parsed exactly."""
    raw, df, _ = pipeline
    raw = raw.set_index(df["source_row"].to_numpy() - 1).loc[df["source_row"] - 1]  # align by source row
    for spec in schema["columns"]:
        r, p = raw[spec["raw"]].to_numpy(), df[spec["name"]].to_numpy()
        kind = spec["type"]
        if kind == "float":
            assert np.array_equal(np.array([float(v) for v in r]), p.astype(float)), spec["name"]
        elif kind == "int":
            assert np.array_equal(np.array([int(v) for v in r]), p.astype("int64")), spec["name"]
        elif kind == "zip":
            assert all(f"{int(a):05d}" == b for a, b in zip(r, p))
        elif kind == "bool":
            assert np.array_equal(r == "yes", p.astype(bool))
        else:
            assert np.array_equal(r, p.astype(object)), spec["name"]


@full
def test_record_ids_are_sequential_and_traceable(pipeline):
    _, df, _ = pipeline
    assert df["record_id"].is_unique
    assert df["record_id"].iloc[0] == "ROW-000001" and df["record_id"].iloc[-1] == "ROW-100000"
    assert (df["record_id"].str[4:].astype(int) == df["source_row"]).all()


@full
def test_quality_flag_counts_match_phase1(pipeline):
    _, _, report = pipeline
    assert report["quality_flags"] == {
        "dq_critical_exceeds_total_alarms": 15_749,
        "dq_availability_at_cap": 11_872,
        "dq_setup_success_at_cap": 10_022,
        "dq_data_usage_at_floor": 19_682,
    }


@full
def test_no_values_were_nulled_or_rows_dropped_on_real_data(pipeline):
    _, df, report = pipeline
    assert int(df.isna().sum().sum()) == 0
    steps = {e["step"]: e["rows_affected"] for e in report["cleaning_log"]
             if e["step"] in ("null_invalid_values", "drop_empty_records", "drop_duplicates")}
    assert sum(steps.values()) == 0


@pytest.mark.skipif(not (RAW_PATH.parent.parent / "processed" / "records.parquet").exists(),
                    reason="processed dataset not built (python -m src.preprocessing.processor)")
def test_saved_parquet_round_trips_with_valid_schema(schema):
    from src.preprocessing.loader import load_processed
    df = load_processed()
    assert df.shape == (100_000, 33)
    assert validate_processed(df, schema).ok
