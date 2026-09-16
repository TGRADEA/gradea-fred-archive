"""Inline the data payload into the dashboard template to produce one self-contained file.

The dashboard has to work from ``file://`` and from a static host with no server,
so it cannot fetch its data at runtime. The payload is injected into the template
instead, and the result is a single HTML file with no external dependency beyond
the webfont.
"""

from __future__ import annotations

import json
from pathlib import Path

from .export import build_payload

TEMPLATE = Path(__file__).resolve().parent.parent / "dashboard" / "template.html"
OUTPUT = Path(__file__).resolve().parent.parent / "dashboard" / "regime-desk.html"
PLACEHOLDER = "__DATA__"


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
