"""US state / territory codes used in the dataset's `State` column (59 codes)."""

from __future__ import annotations

STATE_NAMES: dict[str, str] = {
    "AL": "Alabama", "AK": "Alaska", "AZ": "Arizona", "AR": "Arkansas", "CA": "California",
    "CO": "Colorado", "CT": "Connecticut", "DE": "Delaware", "DC": "District of Columbia",
    "FL": "Florida", "GA": "Georgia", "HI": "Hawaii", "ID": "Idaho", "IL": "Illinois",
    "IN": "Indiana", "IA": "Iowa", "KS": "Kansas", "KY": "Kentucky", "LA": "Louisiana",
    "ME": "Maine", "MD": "Maryland", "MA": "Massachusetts", "MI": "Michigan", "MN": "Minnesota",
    "MS": "Mississippi", "MO": "Missouri", "MT": "Montana", "NE": "Nebraska", "NV": "Nevada",
    "NH": "New Hampshire", "NJ": "New Jersey", "NM": "New Mexico", "NY": "New York",
    "NC": "North Carolina", "ND": "North Dakota", "OH": "Ohio", "OK": "Oklahoma", "OR": "Oregon",
    "PA": "Pennsylvania", "RI": "Rhode Island", "SC": "South Carolina", "SD": "South Dakota",
    "TN": "Tennessee", "TX": "Texas", "UT": "Utah", "VT": "Vermont", "VA": "Virginia",
    "WA": "Washington", "WV": "West Virginia", "WI": "Wisconsin", "WY": "Wyoming",
    # territories / freely associated states present in the data
    "AS": "American Samoa", "FM": "Federated States of Micronesia", "GU": "Guam",
    "MH": "Marshall Islands", "MP": "Northern Mariana Islands", "PR": "Puerto Rico",
    "PW": "Palau", "VI": "U.S. Virgin Islands",
}

_NAME_TO_CODE = {v.casefold(): k for k, v in STATE_NAMES.items()}


def to_state_code(value: str) -> str | None:
    """'Minnesota' / 'minnesota' / 'MN' / 'mn' -> 'MN'; unknown -> None."""
    v = value.strip()
    if v.upper() in STATE_NAMES:
        return v.upper()
    return _NAME_TO_CODE.get(v.casefold())
