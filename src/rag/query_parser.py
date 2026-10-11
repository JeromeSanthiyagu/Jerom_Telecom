"""Query understanding (Phase 13).

Extracts from a natural-language query, deterministically:
  * metric constraints - explicit values ("latency 120 ms", "signal -112 dBm",
    "packet loss above 5%") and qualitative statements ("high latency",
    "declining handover success", "good signal strength", "normal user load");
  * context mentions - season, weather, full US state names.

Each qualitative statement is turned into a direction for the metric:
  adverse    - the metric is in its bad direction  ("high latency", "weak signal")
  favorable  - explicitly fine                     ("good signal", "low latency")
  normal     - typical / mid-range                 ("normal user load")
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field

from src.preprocessing.conditions import load_conditions
from src.utils.geo import STATE_NAMES

# metric -> phrases (longest first matters; matching is done longest-first)
METRIC_PHRASES: dict[str, list[str]] = {
    "latency_ms": ["latency", "delay", "round trip time", "rtt", "response time"],
    "packet_loss_pct": ["packet loss", "packet-loss", "packet drops", "lost packets"],
    "call_drop_pct": ["call drop rate", "call drops", "call drop", "dropped calls", "drop rate"],
    "handover_success_pct": ["handover success rate", "handover success", "handover", "handovers",
                             "handoff", "ho success"],
    "signal_strength_dbm": ["signal strength", "signal quality", "signal", "rsrp", "coverage"],
    "throughput_mbps": ["throughput", "data rate", "bandwidth", "speed"],
    "jitter_ms": ["jitter"],
    "cell_availability_pct": ["cell availability", "availability"],
    "setup_success_pct": ["connection setup success rate", "connection setup success", "setup success",
                          "connection setup", "accessibility", "setup failures", "connection failures"],
    "user_count": ["user count", "user load", "users", "subscribers", "load"],
    "data_usage_gb": ["data usage", "traffic volume", "traffic"],
    "critical_alarm_count": ["critical alarms", "critical alarm count", "critical alarm"],
    "alarm_count": ["alarm count", "alarms"],
    "auth_failures": ["authentication failures", "authentication failure", "auth failures", "login failures"],
    "security_incidents": ["security incidents", "security incident", "security events"],
    "fault_rate_pct": ["fault occurrence rate", "fault occurrence", "fault rate", "faults"],
    "config_success_pct": ["configuration change success", "configuration success", "config success"],
    "parameter_changes": ["parameter changes", "configuration changes", "config changes"],
}

HIGHER = {"high", "higher", "increasing", "increased", "rising", "elevated", "more", "many", "excessive",
          "spiking", "spikes", "frequent", "heavy", "growing", "large", "above", "over", "exceeding"}
LOWER = {"low", "lower", "declining", "decreasing", "decreased", "reduced", "falling", "dropping",
         "fewer", "below", "under", "degrading", "limited", "little"}
BAD = {"poor", "bad", "weak", "degraded", "worse", "worsening", "abnormal", "failing", "unstable"}
GOOD = {"good", "strong", "healthy", "stable", "fine", "acceptable", "excellent", "ok", "okay", "great"}
NORMAL = {"normal", "typical", "average", "moderate", "usual", "nominal", "regular"}

# Mentioning these metrics at all implies a problem ("call drops", "packet loss")
PROBLEM_NOUNS = {"packet_loss_pct", "call_drop_pct", "jitter_ms", "auth_failures", "security_incidents",
                 "critical_alarm_count", "alarm_count", "fault_rate_pct"}

SEASON_WORDS = {"winter": "Winter", "spring": "Spring", "summer": "Summer", "fall": "Fall", "autumn": "Fall"}
WEATHER_WORDS = {"rain": "Rain", "rainy": "Rain", "raining": "Rain", "snow": "Snow", "snowy": "Snow",
                 "snowing": "Snow", "clouds": "Clouds", "cloudy": "Clouds", "overcast": "Clouds",
                 "clear": "Clear", "sunny": "Clear"}

_BOUNDARY = re.compile(r"[,;.+/()]|\band\b|\bor\b|\bbut\b|\bwith\b|\bwhile\b|\bwhereas\b")
_COPULA = {"is", "are", "was", "were", "remains", "remain", "seems", "looks", "getting", "becoming",
           "being", "has", "have", "been", "very", "quite", "too", "still", "also"}

_NUM = r"(-?\d+(?:\.\d+)?)"
_UNIT = r"\s*(?:%|percent|ms|milliseconds|dbm|mbps|gb|hours?|users)?"


@dataclass
class MetricConstraint:
    metric: str
    direction: str | None = None      # adverse | favorable | normal | None
    value: float | None = None
    comparator: str | None = None     # eq | gt | lt
    phrase: str = ""
    span: tuple[int, int] = (0, 0)    # clause span in the lower-cased query (qualifier .. phrase)


@dataclass
class QueryUnderstanding:
    text: str
    constraints: list[MetricConstraint] = field(default_factory=list)
    context: dict[str, str] = field(default_factory=dict)   # season / weather / state
    extra_terms: str = ""                                    # derived-condition labels from supplied metrics

    def add_supplied_metrics(self, metrics: dict) -> None:
        """Merge operator-supplied measurements (API `metrics`) into the query understanding,
        so they steer retrieval exactly like values written in the query text."""
        from src.preprocessing.conditions import evaluate_record

        stated = {c.metric for c in self.constraints}
        for k, v in metrics.items():
            if k in ("season", "weather") and isinstance(v, str):
                self.context.setdefault(k, v)
            elif isinstance(v, (int, float)) and not isinstance(v, bool) and k not in stated:
                self.constraints.append(MetricConstraint(k, None, float(v), "eq", "supplied value"))
        labels = [e["label"].lower() for e in evaluate_record(metrics)["evidence"]]
        self.extra_terms = " ".join(labels)

    @property
    def retrieval_text(self) -> str:
        """Query text for embedding/BM25 with 'good/normal X' clauses removed.

        "good signal strength" would otherwise match records whose text says
        "weak signal strength", because both contain the words "signal strength".
        """
        low = " " + self.text.lower() + " "
        spans = sorted((c.span for c in self.constraints if c.direction in ("favorable", "normal")), reverse=True)
        for a, b in spans:
            low = low[:a] + " " + low[b:]
        return " ".join((low + " " + self.extra_terms).split()) or self.text

    @property
    def has_signal(self) -> bool:
        return bool(self.constraints or self.context)

    def stated_values(self) -> dict[str, float]:
        return {c.metric: c.value for c in self.constraints if c.value is not None and c.comparator == "eq"}

    def adverse_conditions(self) -> list[str]:
        """Conditions the user describes as present (qualitatively or by value)."""
        from src.preprocessing.conditions import evaluate_record

        conds = load_conditions()
        by_metric = {c.metric: n for n, c in conds.items()}
        out = [by_metric[c.metric] for c in self.constraints
               if c.direction == "adverse" and c.value is None and c.metric in by_metric]
        out += evaluate_record(self.stated_values())["conditions"]
        # "above X" / "below X": the condition is certainly present if X is already beyond the threshold
        for c in self.constraints:
            if c.value is None or c.comparator not in ("gt", "lt") or c.metric not in by_metric:
                continue
            cond = conds[by_metric[c.metric]]
            if cond.adverse_high and c.comparator == "gt" and c.value >= cond.threshold:
                out.append(cond.name)
            elif not cond.adverse_high and c.comparator == "lt" and c.value <= cond.threshold:
                out.append(cond.name)
        return sorted(set(out))

    def favorable_conditions(self) -> list[str]:
        """Conditions the user explicitly describes as NOT present (e.g. 'good signal')."""
        by_metric = {c.metric: n for n, c in load_conditions().items()}
        return sorted({by_metric[c.metric] for c in self.constraints
                       if c.direction in ("favorable", "normal") and c.metric in by_metric})

    def to_dict(self) -> dict:
        return {"constraints": [asdict(c) for c in self.constraints], "context": self.context,
                "retrieval_text": self.retrieval_text,
                "adverse_conditions": self.adverse_conditions(),
                "favorable_conditions": self.favorable_conditions()}


def _adverse_high(metric: str) -> bool:
    for c in load_conditions().values():
        if c.metric == metric:
            return c.adverse_high
    return True


def _direction(words: list[str], metric: str) -> str | None:
    for w in reversed(words):
        if w in BAD:
            return "adverse"
        if w in GOOD:
            return "favorable"
        if w in NORMAL:
            return "normal"
        if w in HIGHER:
            return "adverse" if _adverse_high(metric) else "favorable"
        if w in LOWER:
            return "favorable" if _adverse_high(metric) else "adverse"
    return None


def parse_query(text: str) -> QueryUnderstanding:
    q = QueryUnderstanding(text=text)
    low = " " + text.lower() + " "
    taken = [False] * len(low)

    phrases = sorted(((p, m) for m, ps in METRIC_PHRASES.items() for p in ps), key=lambda x: -len(x[0]))
    seen: set[str] = set()
    for phrase, metric in phrases:
        for m in re.finditer(rf"(?<![a-z]){re.escape(phrase)}(?![a-z])", low):
            s, e = m.span()
            if any(taken[s:e]):
                continue
            for i in range(s, e):
                taken[i] = True
            if metric in seen:
                continue
            seen.add(metric)
            # qualifiers only count within the same clause as the metric phrase
            before_seg = _BOUNDARY.split(low[max(0, s - 40):s])[-1]
            before = re.findall(r"[a-z]+", before_seg)[-3:]
            after_txt = low[e:e + 40]
            after_seg = _BOUNDARY.split(after_txt)[0]
            after = [w for w in re.findall(r"[a-z]+", after_seg) if w not in _COPULA][:1]
            direction = _direction(before, metric) or _direction(after, metric)

            value = comparator = None
            num = re.match(rf"\s*(?:is|of|at|=|:|around|about|approximately|of about)?\s*"
                           rf"(above|over|below|under|greater than|less than|more than|>|<)?\s*{_NUM}{_UNIT}",
                           after_txt)
            if num:
                value = float(num.group(2))
                comp = num.group(1)
                comparator = ("gt" if comp in ("above", "over", "greater than", "more than", ">")
                              else "lt" if comp in ("below", "under", "less than", "<") else "eq")
            else:
                pre = re.search(rf"{_NUM}{_UNIT}\s*$", low[max(0, s - 20):s])
                if pre:
                    value, comparator = float(pre.group(1)), "eq"
            if direction is None and value is None and metric in PROBLEM_NOUNS:
                direction = "adverse"
            clause_start = s - len(before_seg)
            q.constraints.append(MetricConstraint(metric, direction, value, comparator, phrase, (clause_start, e)))

    words = re.findall(r"[a-z]+", low)
    for w in words:
        if w in SEASON_WORDS and "season" not in q.context:
            q.context["season"] = SEASON_WORDS[w]
        if w in WEATHER_WORDS and "weather" not in q.context:
            # "clear" is too ambiguous in ordinary English unless "weather"/"sky" is nearby
            if w == "clear" and "weather" not in words and "sky" not in words:
                continue
            q.context["weather"] = WEATHER_WORDS[w]
    for code, name in STATE_NAMES.items():
        if re.search(rf"(?<![a-z]){re.escape(name.lower())}(?![a-z])", low):
            q.context["state"] = code
            break
    return q
