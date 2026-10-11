"""Schema and data-quality validation.

Two stages, two strictness levels:

* `validate_raw`       - before cleaning. Only a missing required column is an
                         error; everything else is a warning the cleaner handles.
* `validate_processed` - after cleaning. Any remaining problem is an error,
                         because downstream phases rely on these guarantees.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import pandas as pd

from src.utils.text import MISSING_TOKENS, collapse_whitespace, normalize_header


class ValidationError(ValueError):
    pass


@dataclass
class ValidationReport:
    stage: str
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    checks: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.errors

    def raise_if_errors(self) -> None:
        if self.errors:
            raise ValidationError(f"{self.stage} validation failed: " + "; ".join(self.errors))

    def to_dict(self) -> dict:
        return {**asdict(self), "ok": self.ok}


def _record_id_pattern(schema: dict) -> str:
    ds = schema["dataset"]
    return rf"{ds['record_id_prefix']}\d{{{ds['record_id_width']}}}"


# --------------------------------------------------------------------------- #
def validate_raw(df: pd.DataFrame, schema: dict) -> ValidationReport:
    """Inspect a raw (all-text) frame against the schema without changing it."""
    rep = ValidationReport("raw")
    specs = {normalize_header(c["raw"]): c for c in schema["columns"]}
    present = {normalize_header(c): c for c in df.columns}

    missing = [specs[k]["raw"] for k in specs if k not in present]
    extra = [present[k] for k in present if k not in specs]
    if missing:
        rep.errors.append(f"missing required columns: {missing}")
    if extra:
        rep.warnings.append(f"unexpected columns (will be dropped): {extra}")

    expected_rows = schema["dataset"].get("expected_rows")
    rep.checks["rows"] = int(len(df))
    rep.checks["columns"] = int(df.shape[1])
    if expected_rows and len(df) != expected_rows:
        rep.warnings.append(f"row count {len(df)} differs from expected {expected_rows}")

    per_col: dict[str, dict] = {}
    for key, spec in specs.items():
        if key not in present:
            continue
        s = df[present[key]].astype("string")
        stripped = s.map(lambda v: collapse_whitespace(v) if isinstance(v, str) else v)
        is_missing = s.isna() | stripped.str.lower().isin(MISSING_TOKENS)
        info = {
            "missing": int(is_missing.sum()),
            "whitespace_issues": int((s.notna() & (s != stripped)).sum()),
        }
        if spec["type"] in ("float", "int", "zip"):
            num = pd.to_numeric(stripped.where(~is_missing), errors="coerce")
            info["unparseable"] = int((~is_missing & num.isna()).sum())
            if "range" in spec:
                lo, hi = spec["range"]
                out = pd.Series(False, index=num.index)
                if lo is not None:
                    out |= (num < lo).fillna(False)
                if hi is not None:
                    out |= (num > hi).fillna(False)
                info["out_of_range"] = int(out.sum())
        if spec["type"] == "category" and "allowed" in spec:
            allowed = {v.casefold() for v in spec["allowed"]} | {
                k.casefold() for k in spec.get("aliases", {})}
            vals = stripped[~is_missing].str.casefold()
            info["unknown_categories"] = int((~vals.isin(allowed)).sum())
        per_col[spec["name"]] = info
        for k, v in info.items():
            if v and k != "missing":
                rep.warnings.append(f"{spec['raw']}: {v} {k.replace('_', ' ')}")
        if info["missing"]:
            rep.warnings.append(f"{spec['raw']}: {info['missing']} missing")

    rep.checks["per_column"] = per_col
    rep.checks["exact_duplicate_rows"] = int(df.duplicated().sum())
    if rep.checks["exact_duplicate_rows"]:
        rep.warnings.append(f"{rep.checks['exact_duplicate_rows']} exact duplicate rows")
    return rep


# --------------------------------------------------------------------------- #
_EXPECTED_DTYPES = {"float": "float64", "int": "Int64", "bool": "boolean",
                    "category": "string", "text": "string", "zip": "string"}


def validate_processed(df: pd.DataFrame, schema: dict) -> ValidationReport:
    """Strict post-cleaning checks. Downstream code may rely on all of these."""
    rep = ValidationReport("processed")
    rep.checks["rows"] = int(len(df))

    # identifiers
    for col in ("record_id", "source_row"):
        if col not in df:
            rep.errors.append(f"missing identifier column {col}")
    if "record_id" in df:
        rid = df["record_id"]
        if rid.isna().any() or rid.duplicated().any():
            rep.errors.append("record_id must be unique and non-null")
        bad = ~rid.astype(str).str.fullmatch(_record_id_pattern(schema))
        if bad.any():
            rep.errors.append(f"{int(bad.sum())} record_id values do not match the format")

    # columns, dtypes, ranges, categories
    missing_counts = {}
    for spec in schema["columns"]:
        col = spec["name"]
        if col not in df:
            rep.errors.append(f"missing column {col}")
            continue
        s = df[col]
        expected = _EXPECTED_DTYPES[spec["type"]]
        if str(s.dtype) != expected:
            rep.errors.append(f"{col}: dtype {s.dtype}, expected {expected}")
        missing_counts[col] = int(s.isna().sum())
        if "range" in spec:
            lo, hi = spec["range"]
            if lo is not None and bool((s < lo).fillna(False).any()):
                rep.errors.append(f"{col}: values below {lo}")
            if hi is not None and bool((s > hi).fillna(False).any()):
                rep.errors.append(f"{col}: values above {hi}")
        if "allowed" in spec:
            bad = s.dropna()[~s.dropna().isin(spec["allowed"])]
            if len(bad):
                rep.errors.append(f"{col}: values outside {spec['allowed']}: {bad.unique()[:3].tolist()}")
        if spec["type"] == "zip":
            bad = s.dropna()[~s.dropna().str.fullmatch(r"\d{5}")]
            if len(bad):
                rep.errors.append(f"{col}: {len(bad)} values are not 5-digit strings")
        if spec["type"] in ("category", "text"):
            vals = s.dropna()
            if (vals != vals.map(collapse_whitespace)).any():
                rep.errors.append(f"{col}: untrimmed whitespace remains")

    rep.checks["missing_per_column"] = missing_counts
    for col, n in missing_counts.items():
        if n:
            rep.warnings.append(f"{col}: {n} missing (left missing, not imputed)")

    data_cols = [c["name"] for c in schema["columns"] if c["name"] in df]
    dups = int(df.duplicated(subset=data_cols).sum())
    rep.checks["duplicate_records"] = dups
    if dups:
        rep.errors.append(f"{dups} duplicate records remain")

    evidence = [c for c in schema["evidence_columns"] if c in df]
    empty = int(df[evidence].isna().all(axis=1).sum())
    if empty:
        rep.errors.append(f"{empty} rows have no network evidence")
    return rep
