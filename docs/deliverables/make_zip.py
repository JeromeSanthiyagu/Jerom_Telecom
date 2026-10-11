"""Create submission/Jerom_Telecom.zip with exactly the files git would include (no secrets, no generated data).

    python -m docs.deliverables.make_zip
"""

from __future__ import annotations

import subprocess
import zipfile

from docs.deliverables.kit import REPO
from src.utils.config import PROJECT_ROOT

OUT = PROJECT_ROOT / "submission" / f"{REPO}.zip"
FORBIDDEN = (".env", "data/raw/", "data/processed/", "data/feedback/", "node_modules/")


def files() -> list[str]:
    out = subprocess.run(["git", "ls-files", "--cached", "--others", "--exclude-standard"], cwd=PROJECT_ROOT,
                         capture_output=True, text=True, check=True).stdout.splitlines()
    return sorted({f for f in out if not f.startswith("submission/")})


def main() -> None:
    names = files()
    bad = [f for f in names if f == ".env" or any(f.startswith(p) or f"/{p}" in f for p in FORBIDDEN[1:])]
    if bad:
        raise SystemExit(f"refusing to package: {bad}")
    OUT.parent.mkdir(exist_ok=True)
    # A deck rebuilt while the original was open in PowerPoint is saved as *.new.pptx; package the newer one.
    pending = PROJECT_ROOT / "presentation" / "Project_Presentation.new.pptx"
    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as z:
        for f in names:
            src = PROJECT_ROOT / f
            if f == "presentation/Project_Presentation.pptx" and pending.exists() and pending.stat().st_mtime > src.stat().st_mtime:
                src = pending
            z.write(src, f"{REPO}/{f}")
    print(f"wrote {OUT} ({len(names)} files, {OUT.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
