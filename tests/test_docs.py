"""Generated documents: regenerate-and-compare, number tracing, rule registry integrity, repository hygiene."""
from __future__ import annotations

import ast
import json
import re
import subprocess

import pytest

from npalloc import numcheck, report
from npalloc.rules_registry import RULES

GENERATED = ["README.md", "docs/rules.md", "docs/decision_brief.md"]


@pytest.fixture(scope="module")
def rendered(root, params):
    return report.render_all(root, params)


@pytest.fixture(scope="module")
def result_numbers(root):
    return numcheck.collect_results(root / "results")


@pytest.mark.parametrize("rel", GENERATED)
def test_generated_document_is_up_to_date(root, rendered, rel):
    on_disk = (root / rel).read_text(encoding="utf-8").replace("\r\n", "\n")
    assert on_disk == rendered[rel], f"{rel} is stale: run python scripts/make_docs.py"


@pytest.mark.parametrize("rel", GENERATED)
def test_every_number_in_document_comes_from_results(root, result_numbers, rel):
    numbers, strings = result_numbers
    text = (root / rel).read_text(encoding="utf-8")
    assert numcheck.unverified(text, numbers, strings) == []
    assert len(numcheck.parse_tokens(numcheck.strip_code(text))) > 10      # the check is not vacuous


def test_readme_first_line_and_sections(root):
    text = (root / "README.md").read_text(encoding="utf-8")
    assert text.splitlines()[0] == "Personal project on public data; not affiliated with any company."
    for heading in ("## Headline results", "## Data and licence", "## Limitations", "## 中文摘要", "## Verification"):
        assert heading in text
    assert "CC BY 4.0" in text and "https://doi.org/10.24432/C5CG6D" in text


def test_brief_is_bilingual(root):
    text = (root / "docs" / "decision_brief.md").read_text(encoding="utf-8")
    assert "## English" in text and "## 中文" in text
    assert len(re.findall(r"[一-鿿]", text)) > 200


# ----------------------------------------------------------------- numcheck


def test_numcheck_catches_a_wrong_number(root, result_numbers):
    numbers, strings = result_numbers
    text = (root / "README.md").read_text(encoding="utf-8")
    fc = json.loads((root / "results" / "forecast.json").read_text(encoding="utf-8"))
    shown = f"{fc['pre_launch']['analogues_k10']['male']:.3f}"
    assert shown in text
    tampered = text.replace(shown, "7.777123", 1)
    assert numcheck.unverified(tampered, numbers, strings) == ["7.777123"]


def test_numcheck_rules():
    numbers, strings = [0.1234, 437.0, 1067371.0, -0.0531], {"2010-06-01", "CC BY 4.0"}
    ok = "Share 12.3% of 437 launches, 1,067,371 rows, 0.123 raw, since 2010-06-01, change -5.3%, +5.3 pp."
    assert numcheck.unverified(ok, numbers, strings) == []
    assert numcheck.unverified("12.3 without a percent sign", numbers, strings) == ["12.3"]
    assert numcheck.unverified("rounded wrongly 0.124", numbers, strings) == ["0.124"]
    assert numcheck.unverified("date 2010-06-02", numbers, strings) == ["2010-06-02"]
    assert numcheck.unverified("438 launches", numbers, strings) == ["438"]
    # identifiers, code spans, fenced blocks and URLs are not claims
    assert numcheck.unverified("rule A1, k10, `99 in code`, https://example.org/502/x\n```\n77\n```", numbers, strings) == []


def test_numcheck_collects_strings_and_nested_numbers(tmp_path):
    (tmp_path / "a.json").write_text(json.dumps({"x": [1.5, {"y": 2}], "s": "CC BY 4.0", "flag": True}), encoding="utf-8")
    numbers, strings = numcheck.collect_results(tmp_path)
    assert sorted(numbers) == [1.5, 2.0, 4.0]
    assert "CC BY 4.0" in strings and "x" in strings


def test_parse_tokens():
    toks = numcheck.parse_tokens("gain +1.25% or 3 pp, 1,234 units")
    assert [(t[1], t[2], t[3]) for t in toks] == [(1.25, 2, True), (3.0, 0, True), (1234.0, 0, False)]


# ----------------------------------------------------------- rule registry


def _test_functions(root):
    found = set()
    for path in (root / "tests").glob("test_*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and node.name.startswith("test_"):
                found.add(f"{path.name}::{node.name}")
    return found


def test_rule_ids_are_unique_and_complete():
    ids = [r["id"] for r in RULES]
    assert len(ids) == len(set(ids))
    for r in RULES:
        assert r["statement"] and r["tests"] and r["affects"], r["id"]


def test_every_rule_points_to_existing_parameters(params):
    for r in RULES:
        for path in r["params"]:
            report.lookup(params, path)         # raises KeyError if the parameter does not exist


def test_every_rule_points_to_existing_tests(root):
    existing = _test_functions(root)
    for r in RULES:
        for t in r["tests"]:
            assert t in existing, f"rule {r['id']} cites a missing test {t}"


def test_every_rule_points_to_existing_results(root):
    cache = {}
    for r in RULES:
        for ref in r["affects"]:
            fname, key = ref.split(":")
            if fname not in cache:
                cache[fname] = json.loads((root / "results" / fname).read_text(encoding="utf-8"))
            report.lookup(cache[fname], key)    # raises KeyError if the result key does not exist


def test_rules_document_lists_every_rule(root):
    text = (root / "docs" / "rules.md").read_text(encoding="utf-8")
    for r in RULES:
        assert f"| {r['id']} |" in text


# ---------------------------------------------------------------- results


def test_results_are_internally_consistent(root):
    fc = json.loads((root / "results" / "forecast.json").read_text(encoding="utf-8"))
    al = json.loads((root / "results" / "allocation.json").read_text(encoding="utf-8"))
    ln = json.loads((root / "results" / "launches.json").read_text(encoding="utf-8"))
    fn = ln["funnel"]
    assert fn["first_sale_in_burn_in"] + fn["first_sale_after_burn_in"] == fn["products"]
    assert fn["first_sale_after_burn_in"] - fn["horizon_not_observed"] == fn["candidates"]
    assert fn["candidates"] - fn["rejected_variant"] - fn["rejected_same_description"] - fn["rejected_succession"] == fn["launches"]
    assert sum(ln["launches_by_month"].values()) == fn["launches"] == sum(ln["launches_by_category"].values())
    assert sum(o["test"] for o in fc["origins"]) == fc["test_launches"] == al["context"]["launches"]
    assert sum(fc["selected_method_launches"].values()) == fc["test_launches"]
    assert al["context"]["demand_units"] == fc["test_units"]
    for name, pol in al["policies"].items():
        assert pol["shipped_units"] + pol["leftover_dc_units"] == pytest.approx(al["context"]["stock_units"]), name
        sold = pol["fill_rate"] * al["context"]["demand_units"]
        assert sold + pol["lost_units"] == pytest.approx(al["context"]["demand_units"]), name
        assert pol["value"] <= al["policies"]["oracle"]["value"] + 1e-6 or "two_stage" in name
    assert al["mip_check"]["objective_agree"] == al["mip_check"]["launches"]
    # every training set ends before its origin: train counts can only grow
    trains = [o["train"] for o in fc["origins"]]
    assert trains == sorted(trains)


def test_per_launch_table_matches_summary(root):
    import pandas as pd

    fc = json.loads((root / "results" / "forecast.json").read_text(encoding="utf-8"))
    al = json.loads((root / "results" / "allocation.json").read_text(encoding="utf-8"))
    t = pd.read_csv(root / "results" / "per_launch.csv", dtype={"code": str})
    launches = pd.read_csv(root / "data" / "derived" / "launches.csv", dtype={"code": str})
    assert len(t) == fc["test_launches"] and t["code"].is_unique
    assert float(t["actual"].sum()) == fc["test_units"]
    day = launches.set_index("code")["launch_day"].reindex(t["code"])
    assert (day.to_numpy() >= t["origin"].to_numpy()).all()             # no test launch precedes its origin
    assert t["value_newsvendor"].sum() == pytest.approx(al["policies"]["newsvendor"]["value"], rel=1e-6)
    assert t["value_proportional"].sum() == pytest.approx(al["policies"]["proportional"]["value"], rel=1e-6)


# ---------------------------------------------------------------- hygiene


def _tracked(root):
    try:
        out = subprocess.run(["git", "ls-files"], cwd=root, capture_output=True, text=True, check=True).stdout.split("\n")
        files = [f for f in out if f]
    except Exception:
        files = []
    if not files:   # not a git checkout (e.g. the mutation work copy): scan the source tree instead
        skip = {".venv", ".git", ".mutation_work", "__pycache__", "raw", "interim"}
        files = [str(p.relative_to(root)).replace("\\", "/") for p in root.rglob("*")
                 if p.is_file() and not (set(p.relative_to(root).parts) & skip)]
    return files


def test_no_local_paths_or_raw_data_in_repository(root):
    drive = re.compile(r"[A-Za-z]:[\\/]+Users[\\/]|/c/Users/|/home/[a-z]|/Users/[A-Za-z]")
    files = _tracked(root)
    assert files
    for rel in files:
        assert not rel.startswith(("data/raw/", "data/interim/")), rel
        assert not rel.lower().endswith((".xlsx", ".zip")), rel
        p = root / rel
        if p.suffix.lower() in {".py", ".md", ".json", ".csv", ".toml", ".txt", ".yml", ".cfg", ""}:
            text = p.read_text(encoding="utf-8", errors="ignore")
            if rel == "tests/test_docs.py":
                continue
            assert not drive.search(text), f"local path in {rel}"
            assert "@" not in text or rel in {"LICENSE", ".github/workflows/tests.yml"} or not re.search(r"[\w.+-]+@[\w-]+\.[\w.]+", text), f"e-mail address in {rel}"


def test_derived_extracts_hold_no_customer_identifiers(root):
    for name in ("launches.csv", "launch_weekly.csv", "channel_month.csv", "candidates_rejected.csv"):
        header = (root / "data" / "derived" / name).read_text(encoding="utf-8").splitlines()[0].lower()
        assert "customer" not in header and "invoice" not in header


def test_licence_sources_and_workflow(root):
    lic = (root / "LICENSE").read_text(encoding="utf-8")
    assert lic.startswith("MIT License") and "LUOaini1213" in lic
    src = (root / "docs" / "sources.md").read_text(encoding="utf-8")
    hashes = json.loads((root / "data" / "hashes.json").read_text(encoding="utf-8"))
    for f in hashes["files"].values():
        assert f["sha256"] in src
    assert hashes["source_url"] in src and "Could not verify" in src and "CC BY 4.0" in src
    wf = (root / ".github" / "workflows" / "tests.yml").read_text(encoding="utf-8")
    assert "pip install -r requirements.txt" in wf and "pytest" in wf
    reqs = [ln for ln in (root / "requirements.txt").read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert reqs and all("==" in ln for ln in reqs)          # every requirement is pinned
    assert ".venv/" in (root / ".gitignore").read_text(encoding="utf-8")
