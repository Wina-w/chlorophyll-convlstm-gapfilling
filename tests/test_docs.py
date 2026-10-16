"""README and docs: links resolve, and the results they quote match the committed outputs/*.json.

The published numbers live in outputs/; these tests keep the prose from drifting away from them
(no data or checkpoint needed).
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DOCS = ["README.md", "docs/RESULTS.md", "docs/MIGRATION.md", "data/README.md"]
RESULTS = [p for p in sorted(ROOT.glob("outputs/**/*.json")) if "dineof_work" not in p.parts]
# quoted from somewhere other than a results JSON, and said so where it is quoted
NOT_IN_JSON = {"0.987"}  # DINEOF's own cross-validation error in the July 2026 SST run's log


def _prose(path: Path) -> str:
    return re.sub(r"^```.*?^```", "", path.read_text(encoding="utf-8"), flags=re.M | re.S)  # no code blocks


def _anchors(path: Path) -> set[str]:
    # GitHub heading ids: lower case, punctuation dropped, spaces to hyphens
    return {
        re.sub(r"[^\w\- ]", "", h.strip().lower()).replace(" ", "-")
        for h in re.findall(r"^#+ (.+)$", _prose(path), re.M)
    }


def _json(name: str) -> dict:
    return json.loads((ROOT / "outputs" / name).read_text(encoding="utf-8"))


def _n(x: float) -> str:  # a score, as the docs print it: "0.109"
    return f"{x:.3f}"


def _s(x: float) -> str:  # a difference: "+0.016", "−0.034" (the docs use the U+2212 minus)
    return f"{x:+.3f}".replace("-", "−")


def _diff(d: dict) -> str:
    return f"{_s(d['point'])} [{_s(d['lo'])}, {_s(d['hi'])}]"


def _verdict(d: dict, if_negative: str, if_positive: str) -> str:
    return "unresolved" if not d["resolved"] else (if_negative if d["point"] < 0 else if_positive)


@pytest.mark.parametrize("doc", DOCS)
def test_doc_links_resolve(doc):
    path = ROOT / doc
    for target in re.findall(r"\]\(([^)\s]+)\)", _prose(path)):
        if target.startswith(("http://", "https://")):
            continue
        file, _, anchor = target.partition("#")
        dest = (path.parent / file).resolve() if file else path
        assert dest.exists(), f"{doc}: broken link {target}"
        if anchor and dest.suffix == ".md":
            assert anchor in _anchors(dest), f"{doc}: no heading for #{anchor} in {dest.name}"
    for ref in re.findall(r"`((?:src|scripts|tests|configs|docs|notebooks)/[^`\s*{]+)`", _prose(path)):
        assert (ROOT / ref.split("::")[0]).exists(), f"{doc}: {ref} does not exist"


def test_every_three_decimal_number_in_the_results_docs_is_a_committed_result():
    values = []

    def walk(x):
        if isinstance(x, dict):
            for v in x.values():
                walk(v)
        elif isinstance(x, list):
            for v in x:
                walk(v)
        elif isinstance(x, float | int) and not isinstance(x, bool):
            values.append(round(abs(float(x)), 3))

    for p in RESULTS:
        walk(json.loads(p.read_text(encoding="utf-8")))
    known = set(values)
    for doc in ("README.md", "docs/RESULTS.md"):
        quoted = re.findall(r"(?<![\w.])[−+-]?(\d+\.\d{3})(?![\w.])", _prose(ROOT / doc))
        missing = sorted({q for q in quoted if round(float(q), 3) not in known} - NOT_IN_JSON)
        assert not missing, f"{doc} quotes numbers found in no outputs/*.json: {missing}"


def test_results_at_a_glance_matches_the_committed_json():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    m, dn, ps = _json("metrics.json"), _json("dineof_vs_ml_chl.json"), _json("per_satellite.json")
    matched, rows = m["log10_space_matched"], []
    for base, verdict_key in (
        ("persistence", "model_vs_persistence"),
        ("interp_fusion", "model_vs_interp_fusion"),
    ):
        d = {
            **m["verdicts"][verdict_key]["log_rmse_diff"],
            "resolved": m["verdicts"][verdict_key]["resolved"],
        }
        rows.append(
            f"| {_n(matched['model']['rmse'])} vs {_n(matched[base]['rmse'])} | {_diff(d)} | "
            + _verdict(d, "ConvLSTM better", f"{base} better")
        )
    for key in ("observed_slice", "independence_slice"):  # model vs persistence, truth = the observation
        pr = m[key]["paired_vs_observation"]
        d = pr["diff"]["model_minus_persistence"]
        rows.append(
            f"| {_n(pr['methods']['model']['rmse']['point'])} vs "
            f"{_n(pr['methods']['persistence']['rmse']['point'])} | {_diff(d)} | "
            + _verdict(d, "ConvLSTM better", "persistence better")
        )
    for key, design in (
        ("all_hole_px_vs_target", "dineof_obs"),
        ("all_hole_px_vs_target", "dineof_target_at_obs"),
        ("observed_px_vs_observation", "dineof_obs"),
    ):
        r = dn[key]
        d = r["diff"][f"{design}_minus_convlstm"]
        rows.append(
            f"| {_n(r['methods'][design]['rmse']['point'])} vs {_n(r['methods']['convlstm']['rmse']['point'])} | "
            f"{_diff(d)} | " + _verdict(d, "DINEOF better", "ConvLSTM better")
        )
    pv = ps["paired_vs_fusion"]
    fusion = _n(pv["methods"]["fusion"]["rmse"]["point"])
    d = pv["diff"]["single:viirs_noaa20_minus_fusion"]
    rows.append(
        f"| {_n(pv['methods']['single:viirs_noaa20']['rmse']['point'])} vs {fusion} | {_diff(d)} | "
        + _verdict(d, "single better", "fusion better")
    )
    a, b = (pv["diff"][f"single:{s}_minus_fusion"] for s in ("modis_aqua", "viirs_snpp"))
    assert a["resolved"] and b["resolved"] and a["point"] > 0 and b["point"] > 0
    rows.append(
        f"| {_n(pv['methods']['single:modis_aqua']['rmse']['point'])} / "
        f"{_n(pv['methods']['single:viirs_snpp']['rmse']['point'])} vs {fusion} | "
        f"{_diff(a)} / {_diff(b)} | fusion better"
    )
    for row in rows:
        assert row in readme, f"README 'Results at a glance' disagrees with outputs/: expected {row!r}"


def test_main_table_matches_metrics_json():
    results = (ROOT / "docs" / "RESULTS.md").read_text(encoding="utf-8")
    m = _json("metrics.json")
    for method, ci in m["log10_space_ci"].items():
        r = ci["rmse"]
        n = f"{ci['n']:,}".replace(",", " ")  # "6 792"
        r2 = f"{ci['r2']['point']:.2f}".replace("-", "−")
        chl = f"{m['chlorophyll_units'][method]['rmse']:#.3g}"  # "0.496", "1.70"
        row = f"| {n} | {_n(r['point'])} [{_n(r['lo'])}, {_n(r['hi'])}] | {r2} | {chl} |"
        assert row in results, f"docs/RESULTS.md main table, {method}: expected {row!r}"
