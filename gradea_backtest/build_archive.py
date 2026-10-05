"""Build the FRED archive page: one self-contained HTML file describing the mirror.

    python -m gradea_backtest.build_archive [--out PATH]

The page shows what this repository holds and whether it is intact: each
series' span, latest value, full history chart, and a SHA-256 check of the CSV
against ``manifest.json``. It is the archive's own status page, so it uses only
the standard library -- the archive needs none of the backtesting runtime, and
neither does its page.

Like the dashboard, the payload is inlined into the template so the result opens
from ``file://``. The output is gitignored for the same reason the dashboard's
is: it is regenerable and changes every time the archive does.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TEMPLATE = ROOT / "archive" / "template.html"
OUTPUT = ROOT / "archive" / "fred-archive.html"
PLACEHOLDER = "__DATA__"

#: Display metadata. Labels match ``data.SERIES``, which cannot be imported here
#: without pulling in numpy and pandas.
SERIES = {
    "BAMLH0A0HYM2": ("ICE BofA US High Yield OAS", "percent", "business-daily"),
    "BAMLC0A0CM": ("ICE BofA US Corporate (IG) OAS", "percent", "business-daily"),
    "DGS10": ("10-Year Treasury CMT", "percent", "business-daily"),
    "DGS2": ("2-Year Treasury CMT", "percent", "business-daily"),
    "NFCI": ("Chicago Fed National Financial Conditions Index", "index", "weekly"),
}

#: Series FRED serves only on a rolling ~3-year window. Anything older than the
#: window exists nowhere but this archive.
WINDOWED = ("BAMLH0A0HYM2", "BAMLC0A0CM")


def _read_csv(path: Path) -> tuple[list[str], list[float]]:
    dates: list[str] = []
    values: list[float] = []
    with path.open(newline="") as fh:
        reader = csv.reader(fh)
        header = next(reader)
        if header != ["observation_date", "value"]:
            raise ValueError(f"{path} has unexpected header {header!r}")
        for row in reader:
            dates.append(row[0])
            values.append(float(row[1]))
    return dates, values


def _git_history(path: Path) -> list[dict]:
    """Commits touching one CSV, newest first. Empty outside a git checkout."""
    try:
        out = subprocess.run(
            ["git", "log", "--format=%h%x09%aI%x09%s", "--", path.name],
            cwd=ROOT, capture_output=True, text=True, check=True, timeout=20,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    commits = []
    for line in out.splitlines():
        sha, when, subject = line.split("\t", 2)
        commits.append({"sha": sha, "date": when[:10], "subject": subject})
    return commits


def build_payload(root: Path = ROOT) -> dict:
    manifest = json.loads((root / "manifest.json").read_text())
    series = []
    for sid, meta in manifest["series"].items():
        path = root / f"{sid}.csv"
        label, units, freq = SERIES.get(sid, (sid, "", ""))
        entry = {
            "id": sid, "label": label, "units": units, "frequency": freq,
            "windowed": sid in WINDOWED, "manifest": meta,
        }
        if not path.exists():
            entry.update(status="missing", sha256=None, dates=[], values=[], history=[])
        else:
            raw = path.read_bytes()
            digest = hashlib.sha256(raw).hexdigest()
            dates, values = _read_csv(path)
            entry.update(
                status="ok" if digest == meta["sha256"] else "mismatch",
                sha256=digest, bytes=len(raw), dates=dates, values=values,
                history=_git_history(path),
            )
        series.append(entry)
    return {
        "built_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "manifest_updated_at": manifest.get("updated_at"),
        "series": series,
    }


def build(out: Path | str | None = None, payload: dict | None = None) -> Path:
    payload = payload if payload is not None else build_payload()
    html = TEMPLATE.read_text()
    if PLACEHOLDER not in html:
        raise ValueError(f"{TEMPLATE} has no {PLACEHOLDER} placeholder")
    # `</script>` inside a JSON string would close the host <script> tag early.
    blob = json.dumps(payload, separators=(",", ":")).replace("</", "<\\/")
    out = Path(out) if out else OUTPUT
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html.replace(PLACEHOLDER, blob))
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", help=f"output path (default {OUTPUT.relative_to(ROOT)})")
    args = parser.parse_args(argv)
    payload = build_payload()
    out = build(args.out, payload)
    bad = [s["id"] for s in payload["series"] if s["status"] != "ok"]
    print(f"\nwrote {out}  ({out.stat().st_size / 1024:.0f} KB)")
    print(f"  {len(payload['series']) - len(bad)}/{len(payload['series'])} series verified"
          + (f"; FAILED: {', '.join(bad)}" if bad else "") + "\n")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
