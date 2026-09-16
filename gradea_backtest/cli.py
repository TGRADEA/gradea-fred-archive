"""Command-line interface.

    python -m gradea_backtest --help
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

from .data import ARCHIVE_ROOT, SERIES, derived_columns, load_full_panel
from .engine import compare, run_backtest
from .export import build_payload, write_payload
from .ingest import RECOMMENDED, ingest
from .metrics import compute_metrics
from .regimes import FAMILY_DESCRIPTIONS, LABEL_ORDER, build_regimes
from .returns import ASSET_LABELS, build_asset_returns
from .strategies import REGISTRY, available_strategies


def _pct(x, digits=2):
    return "n/a" if x is None or pd.isna(x) else f"{x * 100:.{digits}f}%"


def _num(x, digits=2):
    return "n/a" if x is None or pd.isna(x) else f"{x:.{digits}f}"


def _context():
    panel = load_full_panel()
    return panel, build_asset_returns(panel.raw), build_regimes(panel.pit)


def cmd_list(args) -> int:
    panel, _, _ = _context()
    usable = {s.key for s in available_strategies(panel.pit)}
    print("STRATEGIES\n")
    family = None
    for spec in REGISTRY.values():
        if spec.family != family:
            family = spec.family
            print(f"  [{family}]")
        mark = " " if spec.key in usable else "!"
        print(f"  {mark} {spec.key:20s} {spec.name}")
        if spec.params:
            knobs = ", ".join(f"{p.key}={p.default:g}" for p in spec.params)
            print(f"      params: {knobs}")
    if any(s.key not in usable for s in REGISTRY.values()):
        print("\n  ! = required series not present in the panel")
    print("\nREGIME FAMILIES\n")
    for name, labels in LABEL_ORDER.items():
        print(f"  {name:18s} {' | '.join(labels)}")
    return 0


def _parse_params(pairs: list[str]) -> dict[str, float]:
    out = {}
    for pair in pairs or []:
        if "=" not in pair:
            raise SystemExit(f"--param expects key=value, got {pair!r}")
        k, v = pair.split("=", 1)
        out[k.strip()] = float(v)
    return out


def cmd_run(args) -> int:
    panel, returns, regimes = _context()
    spec = REGISTRY.get(args.strategy)
    if spec is None:
        raise SystemExit(f"unknown strategy {args.strategy!r}; try `list`")
    params = spec.defaults() | _parse_params(args.param)
    weights = spec.build(panel.pit, regimes, **params)
    result = run_backtest(
        spec.key, weights, returns, regimes, description=spec.description,
        start=args.start, end=args.end,
    )
    m = result.metrics

    print(f"\n{spec.name}  [{spec.key}]")
    print(f"{spec.description}\n")
    if params:
        print("params: " + ", ".join(f"{k}={v:g}" for k, v in params.items()) + "\n")
    print(f"  window        {m.start} .. {m.end}  ({m.n_days} days, {m.n_days/252:.1f}y)")
    print(f"  CAGR          {_pct(m.cagr)}     (gross {_pct(result.gross_metrics.cagr)})")
    print(f"  volatility    {_pct(m.ann_vol)}")
    print(f"  Sharpe        {_num(m.sharpe)}      Sortino {_num(m.sortino)}")
    print(f"  max drawdown  {_pct(m.max_drawdown, 1)}     Calmar {_num(m.calmar)}")
    print(f"  hit rate      {_pct(m.hit_rate, 1)}")
    print(f"  turnover      {_num(m.ann_turnover, 1)}x/yr   cost drag "
          f"{_pct(result.gross_metrics.cagr - m.cagr, 3)}")
    if spec.caveat:
        print(f"\n  CAVEAT: {spec.caveat}")
    for w in result.warnings:
        print(f"\n  WARNING: {w}")

    if args.attribution:
        for fam, table in result.attribution.items():
            print(f"\n  by {fam}:")
            for label in LABEL_ORDER.get(fam, list(table.index)):
                if label not in table.index:
                    continue
                r = table.loc[label]
                thin = "  (thin)" if r["thin"] else ""
                print(f"    {label:18s} ret {_pct(r['ann_return']):>8s}  vol {_pct(r['ann_vol']):>7s}"
                      f"  SR {_num(r['sharpe']):>6s}  {int(r['n_days']):5d}d{thin}")
    print()
    return 0


def cmd_compare(args) -> int:
    panel, returns, regimes = _context()
    results = []
    for spec in available_strategies(panel.pit):
        weights = spec.build(panel.pit, regimes, **spec.defaults())
        results.append(run_backtest(spec.key, weights, returns, regimes,
                                    start=args.start, end=args.end))
    table = compare(results)
    fmt = table.assign(
        cagr=lambda d: (d.cagr * 100).round(2), vol=lambda d: (d.vol * 100).round(2),
        sharpe=lambda d: d.sharpe.round(2), sortino=lambda d: d.sortino.round(2),
        max_dd=lambda d: (d.max_dd * 100).round(1), calmar=lambda d: d.calmar.round(2),
        hit=lambda d: (d.hit * 100).round(1), ann_turnover=lambda d: d.ann_turnover.round(1),
        cost_drag=lambda d: (d.cost_drag * 1e4).round(1),
    ).rename(columns={"cost_drag": "cost_bp"})
    print("\n" + fmt.to_string() + "\n")
    print("Sharpe is vs a zero cash rate -- see `notes`.\n")
    return 0


def cmd_regimes(args) -> int:
    panel, _, regimes = _context()
    for family in regimes.columns:
        col = regimes[family]
        labelled = int(col.notna().sum())
        first = col.first_valid_index()
        print(f"\n{family}  ({labelled} labelled days from {first.date() if first is not None else 'n/a'})")
        print(f"  {FAMILY_DESCRIPTIONS.get(family, '')}")
        counts = col.value_counts()
        for label in LABEL_ORDER.get(family, list(counts.index)):
            n = int(counts.get(label, 0))
            share = n / labelled if labelled else 0
            bar = "#" * int(round(share * 40))
            print(f"    {label:18s} {n:6d}  {share*100:5.1f}%  {bar}")
    print()
    return 0


def cmd_verify(args) -> int:
    """Check the archive CSVs against manifest.json without modifying anything."""
    manifest = json.loads((ARCHIVE_ROOT / "manifest.json").read_text())
    failures = 0
    print()
    for sid, meta in manifest["series"].items():
        path = ARCHIVE_ROOT / f"{sid}.csv"
        if not path.exists():
            print(f"  MISSING  {sid}")
            failures += 1
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        ok = digest == meta["sha256"]
        failures += 0 if ok else 1
        print(f"  {'ok     ' if ok else 'MISMATCH'} {sid:14s} {meta['obs_count']:6d} obs  "
              f"{meta['first_date']} .. {meta['last_date']}")
        if not ok:
            print(f"           expected {meta['sha256'][:16]}..., got {digest[:16]}...")
    print(f"\n  {len(manifest['series']) - failures}/{len(manifest['series'])} series verified\n")
    return 1 if failures else 0


def cmd_ingest(args) -> int:
    if args.list:
        print("\nRecommended additions, highest value first:\n")
        for r in sorted(RECOMMENDED, key=lambda x: (x.priority, x.series_id)):
            print(f"  P{r.priority}  {r.series_id:11s} {r.label}  (from {r.starts})")
            print(f"      {r.unlocks}\n")
        return 0
    report = ingest(args.series or None)
    print("\n" + report.to_string() + "\n")
    if (report["status"] == "failed").any():
        print("Failures are usually an egress policy blocking fred.stlouisfed.org.")
        print("Everything else in this toolkit runs without it.\n")
        return 1
    return 0


def cmd_export(args) -> int:
    payload = build_payload()
    out = write_payload(args.out, payload)
    size = out.stat().st_size / 1024
    print(f"\nwrote {out}  ({size:.0f} KB)")
    print(f"  {len(payload['strategies'])} strategies, {len(payload['regimes'])} regime families, "
          f"{len(payload['dates'])} sampled dates\n")
    return 0


def cmd_dashboard(args) -> int:
    from .build_dashboard import build

    out = build(args.out)
    print(f"\nwrote {out}  ({out.stat().st_size / 1024:.0f} KB)")
    print("  self-contained -- open it directly in a browser\n")
    return 0


def cmd_notes(args) -> int:
    payload_caveats = build_payload.__doc__
    print("\nMODELLING NOTES\n")
    for i, note in enumerate(
        [
            "Sharpe is computed against a ZERO cash rate. The archive holds no "
            "short-rate series, so every Sharpe on an unlevered bond position is a "
            "total-return Sharpe and is too high. `ingest DFF` fixes this.",
            "Returns are modelled from yields, not observed from prices. Par-bond "
            "repricing with duration and convexity recomputed daily; roll-down is "
            "omitted (understates steep-curve returns), defaults are omitted "
            "(overstates high yield).",
            "The steepener is financed at the 2y yield, giving the textbook "
            "-(slope)/252 + D10*d(slope). Without that financing leg the trade would "
            "appear to earn the yield level for taking no duration risk.",
            "Credit series begin 2023-08-01. Three years, no default cycle.",
            "Signals use publication-lagged data and are held one further day before "
            "trading. NFCI's lag is four business days.",
            "Percentiles and z-scores are expanding, never full-sample.",
            "Strategy defaults were not tuned on this archive. Sweeping the parameter "
            "controls spends the sample's power to tell you anything.",
        ],
        start=1,
    ):
        print(f"  {i}. {note}\n")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="gradea_backtest",
        description="Regime-aware backtesting over the GradeA FRED archive.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("list", help="list strategies and regime families").set_defaults(func=cmd_list)

    p_run = sub.add_parser("run", help="run one strategy")
    p_run.add_argument("strategy")
    p_run.add_argument("--param", action="append", metavar="KEY=VALUE")
    p_run.add_argument("--start")
    p_run.add_argument("--end")
    p_run.add_argument("--attribution", action="store_true", help="break returns down by regime")
    p_run.set_defaults(func=cmd_run)

    p_cmp = sub.add_parser("compare", help="run every strategy side by side")
    p_cmp.add_argument("--start")
    p_cmp.add_argument("--end")
    p_cmp.set_defaults(func=cmd_compare)

    sub.add_parser("regimes", help="regime distributions").set_defaults(func=cmd_regimes)
    sub.add_parser("verify", help="check archive CSVs against manifest.json").set_defaults(func=cmd_verify)
    sub.add_parser("notes", help="modelling assumptions and their costs").set_defaults(func=cmd_notes)

    p_ing = sub.add_parser("ingest", help="fetch additional FRED series into data/extended/")
    p_ing.add_argument("series", nargs="*")
    p_ing.add_argument("--list", action="store_true", help="show recommendations without fetching")
    p_ing.set_defaults(func=cmd_ingest)

    p_dash = sub.add_parser("dashboard", help="build the self-contained dashboard HTML")
    p_dash.add_argument("--out", default=None)
    p_dash.set_defaults(func=cmd_dashboard)

    p_exp = sub.add_parser("export", help="write the dashboard data payload")
    p_exp.add_argument("--out", default="dashboard/data.json")
    p_exp.set_defaults(func=cmd_export)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
