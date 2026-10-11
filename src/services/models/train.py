"""Train all models (Phases 19-22).

    python -m src.services.models.train
"""

from __future__ import annotations

import logging
import time

from src.services.models import fault_risk, incident_classifier, performance
from src.preprocessing.build_incidents import load_incidents


def main() -> dict:
    df = load_incidents()
    out = {}
    for name, mod in [("fault_risk", fault_risk), ("performance", performance), ("classifier", incident_classifier)]:
        t = time.time()
        out[name] = mod.train(df)
        logging.info("%s trained in %.1fs", name, time.time() - t)
    return out


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    main()
