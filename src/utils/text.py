"""Small text-normalisation helpers shared by the validator and the cleaner."""

from __future__ import annotations

import re
import unicodedata

# Strings that mean "no value" in CSV exports. Compared after strip + lower.
MISSING_TOKENS = frozenset({"", "na", "n/a", "nan", "null", "none", "-", "?"})

_WS = re.compile(r"\s+")


def collapse_whitespace(value: str) -> str:
    return _WS.sub(" ", value).strip()


def normalize_header(name: str) -> str:
    """Canonical form of a column header for matching against the schema.

    Handles case, repeated/odd whitespace, Unicode variants and the common
    UTF-8-read-as-Latin-1 artefact of the degree sign ("Â°" -> "°").
    """
    name = name.replace("Â°", "°").replace("﻿", "")
    name = unicodedata.normalize("NFKC", name)
    return collapse_whitespace(name).casefold()
