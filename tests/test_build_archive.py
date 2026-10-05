"""The archive page must report the archive's integrity truthfully."""

from __future__ import annotations

import json
import shutil

from gradea_backtest.build_archive import ROOT, build, build_payload


def test_payload_verifies_every_series_in_the_manifest():
    payload = build_payload()
    manifest = json.loads((ROOT / "manifest.json").read_text())
    assert [s["id"] for s in payload["series"]] == list(manifest["series"])
    for s in payload["series"]:
        assert s["status"] == "ok", s["id"]
        assert len(s["dates"]) == len(s["values"]) == s["manifest"]["obs_count"]
        assert s["dates"][0] == s["manifest"]["first_date"]
        assert s["dates"][-1] == s["manifest"]["last_date"]


def test_tampered_csv_is_reported_as_mismatch(tmp_path):
    for name in ["manifest.json", *(f"{sid}.csv" for sid in ("DGS2", "NFCI"))]:
        shutil.copy(ROOT / name, tmp_path / name)
    (tmp_path / "DGS2.csv").write_bytes((tmp_path / "DGS2.csv").read_bytes().replace(b"7.26", b"7.27", 1))
    status = {s["id"]: s["status"] for s in build_payload(tmp_path)["series"]}
    assert status["DGS2"] == "mismatch"
    assert status["NFCI"] == "ok"
    assert status["DGS10"] == "missing"


def test_build_inlines_payload_without_closing_the_script_tag(tmp_path):
    out = build(tmp_path / "page.html", payload={"series": [], "note": "</script>"})
    html = out.read_text()
    assert "__DATA__" not in html
    assert "<\\/script>" in html
