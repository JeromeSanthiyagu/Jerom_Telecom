"""Cleaning steps for the raw 5G NetOps data.

Guiding rule: the raw dataset is the source of truth. Cleaning may *remove*
an unusable value (set it to missing) or *standardise its representation*
(type, spelling, case), but it never invents, imputes, clips or rounds a
measurement. Every step records what it did and why in a `CleaningLog`.

Steps (applied in this order by `src.preprocessing.processor`):
    1. map_column_names      raw headers -> canonical snake_case names
    2. normalize_text        trim/collapse whitespace, missing tokens -> NA
    3. standardize_categories  case/alias normalisation, unknown -> NA
    4. convert_types         text -> float / Int64 / bool / zero-padded zip
    5. null_invalid_values   physically impossible values -> NA
    6. drop_empty_records    rows with no network evidence at all
    7. drop_duplicates       exact duplicate measurements
    8. add_quality_flags     dq_* annotations (no value is changed)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from src.utils.text import MISSING_TOKENS, collapse_whitespace, normalize_header

ID_COLUMNS = ["record_id", "source_row"]


class SchemaError(ValueError):
    """The input does not have the columns the schema requires."""


@dataclass
class CleaningLog:
    entries: list[dict[str, Any]] = field(default_factory=list)

    def add(self, step: str, action: str, rows: int, reason: str,
            column: str | None = None, examples: list | None = None) -> None:
        self.entries.append({
            "step": step,
            "column": column,
            "action": action,
            "rows_affected": int(rows),
            "reason": reason,
            "examples": examples or [],
        })

    def total(self, step: str) -> int:
        return sum(e["rows_affected"] for e in self.entries if e["step"] == step)


def _columns(schema: dict) -> list[dict]:
    return schema["columns"]


def _examples(values: pd.Series, n: int = 3) -> list:
    return [v.item() if hasattr(v, "item") else v for v in values.drop_duplicates().head(n)]


# --------------------------------------------------------------------------- #
# 1. Column names
# --------------------------------------------------------------------------- #
def map_column_names(df: pd.DataFrame, schema: dict, log: CleaningLog) -> pd.DataFrame:
    """Rename raw headers to canonical names, matching case/whitespace-insensitively."""
    wanted = {normalize_header(c["raw"]): c["name"] for c in _columns(schema)}
    rename, unknown = {}, []
    for col in df.columns:
        if col in ID_COLUMNS:
            continue
        key = normalize_header(col)
        if key in wanted:
            rename[col] = wanted[key]
        elif col in wanted.values():  # already canonical
            rename[col] = col
        else:
            unknown.append(col)

    missing = sorted(set(wanted.values()) - set(rename.values()))
    if missing:
        raise SchemaError(f"Required columns not found: {missing}")
    if len(set(rename.values())) != len(rename):
        raise SchemaError("Two input columns map to the same canonical column.")

    out = df.rename(columns=rename)
    if unknown:
        out = out.drop(columns=unknown)
        log.add("map_column_names", "dropped unknown columns", 0,
                "Not part of the documented schema; kept out of the processed data.",
                examples=unknown)
    changed = [k for k, v in rename.items() if k != v]
    log.add("map_column_names", f"renamed {len(changed)} columns to snake_case", 0,
            "One consistent, code-friendly name per field; the raw->canonical mapping "
            "is in src/config/schema.yaml and docs/DATA_DICTIONARY.md.")
    return out


# --------------------------------------------------------------------------- #
# 2. Text normalisation
# --------------------------------------------------------------------------- #
def normalize_text(df: pd.DataFrame, schema: dict, log: CleaningLog) -> pd.DataFrame:
    out = df.copy()
    for spec in _columns(schema):
        col = spec["name"]
        s = out[col]
        if not (pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s)):
            continue
        present = s.notna()
        collapsed = s[present].astype(str).map(collapse_whitespace)
        changed = int((collapsed != s[present]).sum())
        is_missing = collapsed.str.lower().isin(MISSING_TOKENS)
        collapsed[is_missing] = pd.NA
        out.loc[present, col] = collapsed
        if changed:
            log.add("normalize_text", "trimmed/collapsed whitespace", changed,
                    "Whitespace differences must not create different values.", col)
        if int(is_missing.sum()):
            log.add("normalize_text", "missing tokens -> NA", int(is_missing.sum()),
                    f"Tokens {sorted(MISSING_TOKENS)} mean 'no value' and must not be "
                    "parsed as data.", col, _examples(s[present][is_missing.values]))
    return out


# --------------------------------------------------------------------------- #
# 3. Categories
# --------------------------------------------------------------------------- #
def standardize_categories(df: pd.DataFrame, schema: dict, log: CleaningLog) -> pd.DataFrame:
    out = df.copy()
    for spec in _columns(schema):
        col = spec["name"]
        if spec["type"] != "category":
            continue
        s = out[col]
        present = s.notna()

        if "allowed" in spec:
            lookup = {v.casefold(): v for v in spec["allowed"]}
            lookup.update({k.casefold(): v for k, v in spec.get("aliases", {}).items()})
            mapped = s[present].str.casefold().map(lookup)
            recased = int(((mapped != s[present]) & mapped.notna()).sum())
            invalid = mapped.isna()
            if recased:
                log.add("standardize_categories", "case/alias normalised", recased,
                        f"Canonical spelling is one of {spec['allowed']}.", col)
            if int(invalid.sum()):
                log.add("standardize_categories", "unknown category -> NA", int(invalid.sum()),
                        "Value is not a documented category; guessing would invent data.",
                        col, _examples(s[present][invalid.values]))
            out.loc[present, col] = mapped
        elif spec.get("case") == "upper":
            upper = s[present].str.upper()
            recased = int((upper != s[present]).sum())
            invalid = ~upper.str.fullmatch(r"[A-Z]{2}")
            upper[invalid] = pd.NA
            if recased:
                log.add("standardize_categories", "upper-cased", recased,
                        "State codes are 2-letter upper-case.", col)
            if int(invalid.sum()):
                log.add("standardize_categories", "invalid state code -> NA", int(invalid.sum()),
                        "Not a 2-letter code.", col, _examples(s[present][invalid.values]))
            out.loc[present, col] = upper
    return out


# --------------------------------------------------------------------------- #
# 4. Types
# --------------------------------------------------------------------------- #
def _to_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def _parse_numbers(s: pd.Series) -> pd.Series:
    """Text -> float64 using Python's correctly rounded `float()`.

    `pd.to_numeric` uses a fast parser that can differ from the true value by
    one unit in the last place; Python's parser reproduces the CSV text exactly.
    """
    if pd.api.types.is_numeric_dtype(s):
        return s.astype("float64")
    return s.map(_to_float).astype("float64")



def convert_types(df: pd.DataFrame, schema: dict, log: CleaningLog) -> pd.DataFrame:
    out = df.copy()
    for spec in _columns(schema):
        col, kind = spec["name"], spec["type"]
        s = out[col]
        present = s.notna()

        if kind in ("float", "int", "zip"):
            num = _parse_numbers(s)
            bad = present & num.isna()
            if int(bad.sum()):
                log.add("convert_types", "unparseable number -> NA", int(bad.sum()),
                        "Text that is not a number cannot be a measurement.", col,
                        _examples(s[bad]))
            if kind in ("int", "zip"):
                frac = num.notna() & (num % 1 != 0)
                if int(frac.sum()):
                    log.add("convert_types", "non-integer count -> NA", int(frac.sum()),
                            "A count/identifier cannot be fractional; rounding would "
                            "invent a value.", col, _examples(s[frac]))
                num = num.mask(frac)
            if kind == "float":
                out[col] = num.astype("float64")
            elif kind == "int":
                out[col] = num.astype("Int64")
            else:  # zip: identifier stored as 5-character text
                bad_zip = num.notna() & ((num < 0) | (num > 99999))
                if int(bad_zip.sum()):
                    log.add("convert_types", "invalid zip -> NA", int(bad_zip.sum()),
                            "US ZIP codes have at most 5 digits.", col, _examples(s[bad_zip]))
                num = num.mask(bad_zip)
                out[col] = num.map(lambda v: f"{int(v):05d}" if pd.notna(v) else pd.NA).astype("string")
                log.add("convert_types", "zip stored as 5-character text", int(num.notna().sum()),
                        "ZIP is an identifier, not a quantity; the CSV dropped leading "
                        "zeros (e.g. 503 -> '00503').", col)

        elif kind == "bool":
            truthy = {v.casefold() for v in spec["true_values"]}
            falsy = {v.casefold() for v in spec["false_values"]}
            low = s.str.casefold()
            mapped = pd.Series(pd.NA, index=s.index, dtype="boolean")
            mapped[low.isin(truthy)] = True
            mapped[low.isin(falsy)] = False
            bad = present & mapped.isna()
            if int(bad.sum()):
                log.add("convert_types", "unrecognised yes/no -> NA", int(bad.sum()),
                        "Only documented yes/no spellings are accepted.", col, _examples(s[bad]))
            out[col] = mapped
            log.add("convert_types", "yes/no -> boolean", int(mapped.notna().sum()),
                    "Boolean flag is unambiguous for filtering and modelling.", col)

        else:  # category / text
            out[col] = s.astype("string")
    return out


# --------------------------------------------------------------------------- #
# 5. Impossible values
# --------------------------------------------------------------------------- #
def null_invalid_values(df: pd.DataFrame, schema: dict, log: CleaningLog) -> pd.DataFrame:
    """Set physically impossible values to NA. Valid extremes are untouched."""
    out = df.copy()
    for spec in _columns(schema):
        if "range" not in spec:
            continue
        col = spec["name"]
        lo, hi = spec["range"]
        s = out[col]
        bad = pd.Series(False, index=s.index)
        if lo is not None:
            bad |= (s < lo).fillna(False)
        if hi is not None:
            bad |= (s > hi).fillna(False)
        bad = bad.astype(bool)
        # inf is never a valid measurement
        if pd.api.types.is_float_dtype(s):
            bad |= np.isinf(s.to_numpy(dtype="float64", na_value=np.nan))
        if int(bad.sum()):
            log.add("null_invalid_values", f"outside valid range {spec['range']} -> NA",
                    int(bad.sum()),
                    "Physically/logically impossible; clipping or imputing would "
                    "fabricate a measurement.", col, _examples(s[bad]))
            out[col] = s.mask(bad)
    return out


# --------------------------------------------------------------------------- #
# 6-7. Row-level steps
# --------------------------------------------------------------------------- #
def drop_empty_records(df: pd.DataFrame, schema: dict, log: CleaningLog) -> pd.DataFrame:
    evidence = schema["evidence_columns"]
    empty = df[evidence].isna().all(axis=1)
    if int(empty.sum()):
        log.add("drop_empty_records", "dropped rows with no network evidence", int(empty.sum()),
                f"All of {evidence} are missing, so the row cannot support any analysis.",
                examples=df.loc[empty, "record_id"].head(3).tolist() if "record_id" in df else None)
    return df.loc[~empty]


def drop_duplicates(df: pd.DataFrame, log: CleaningLog) -> pd.DataFrame:
    """Drop exact duplicates of every data column (IDs ignored), keeping the first."""
    data_cols = [c for c in df.columns if c not in ID_COLUMNS]
    dup = df.duplicated(subset=data_cols, keep="first")
    log.add("drop_duplicates", "dropped exact duplicate records", int(dup.sum()),
            "An identical record repeated would be double-counted in retrieval "
            "statistics and leak between ML train/test splits. The first occurrence "
            "keeps its record_id.",
            examples=df.loc[dup, "record_id"].head(3).tolist() if "record_id" in df else None)
    return df.loc[~dup]


# --------------------------------------------------------------------------- #
# 8. Data-quality flags
# --------------------------------------------------------------------------- #
def add_quality_flags(df: pd.DataFrame, schema: dict, log: CleaningLog) -> pd.DataFrame:
    out = df.copy()
    for name, expr in schema.get("quality_flags", {}).items():
        flag = out.eval(expr)
        out[name] = pd.Series(flag, index=out.index).fillna(False).astype(bool)
        log.add("add_quality_flags", f"flag {name}: {expr}", int(out[name].sum()),
                "Annotates a suspicious or capped value for transparency; the value "
                "itself is kept unchanged.")
    return out
