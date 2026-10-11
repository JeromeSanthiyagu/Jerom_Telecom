"""Evidence provenance (Phase 16).

Every important statement in an analysis is stored as a `Claim` with the
kind of source it rests on. The spec's six source types are used, plus
USER_INPUT for observations stated by the operator in the query: labelling
those RAW_DATA would wrongly present them as dataset facts.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class SourceType(str, Enum):
    RAW_DATA = "RAW_DATA"                          # a value present in the dataset
    DERIVED_FEATURE = "DERIVED_FEATURE"            # documented calculation from raw values
    HISTORICAL_EVIDENCE = "HISTORICAL_EVIDENCE"    # statistic over actual retrieved/dataset records
    ML_PREDICTION = "ML_PREDICTION"                # output of a trained model
    ENGINEERING_HYPOTHESIS = "ENGINEERING_HYPOTHESIS"
    RECOMMENDATION = "RECOMMENDATION"
    USER_INPUT = "USER_INPUT"                      # stated by the operator in the query


class Claim(BaseModel):
    claim: str
    source_type: SourceType
    evidence: Any = Field(default=None)


class ProvenanceLog(BaseModel):
    claims: list[Claim] = Field(default_factory=list)

    def add(self, claim: str, source_type: SourceType, evidence: Any = None) -> None:
        self.claims.append(Claim(claim=claim, source_type=source_type, evidence=evidence))

    def by_type(self, source_type: SourceType) -> list[Claim]:
        return [c for c in self.claims if c.source_type == source_type]
