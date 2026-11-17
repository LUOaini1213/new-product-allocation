"""Render README.md, docs/rules.md and docs/decision_brief.md from results/*.json.

Every number in the three documents is read from the result files here;
tests/test_docs.py regenerates the documents, compares them with the committed
files and traces every printed number back to the results.
"""
from __future__ import annotations

from pathlib import Path

from .io import read_json
from .rules_registry import RULES

FIRST_LINE = "Personal project on public data; not affiliated with any company."

METHOD_LABEL = {
    "global_mean_curve": "Global mean launch curve",
    "global_median": "Global median",
    "category_mean": "Category mean curve",
    "category_median": "Category median",
    "nearest_analogue": "Nearest analogue (text + price)",
    "analogues_k10": "Median of nearest analogues",
    "gradient_boosting": "Gradient boosting (attributes)",
    "gradient_boosting_calendar": "Gradient boosting + calendar features",
    "selected": "Selected per origin (rule E2)",
    "selected_random_cv": "Selected by random-fold CV (comparison)",
}
POST_LABEL = {
    "prelaunch_only": "Pre-launch forecast minus sales (no update)",
    "run_rate": "Run rate",
    "curve_ratio": "Curve ratio",
    "bass": "Bass, scale from sales",
    "bass_bayes": "Bass, scale from sales + prior",
}
POLICY_LABEL = {
    "proportional": "Proportional to history",
    "newsvendor_ship_all": "Newsvendor split, ship everything",
    "newsvendor": "Newsvendor, may hold back",
    "proportional_two_stage": "Proportional, two-stage",
    "newsvendor_two_stage": "Newsvendor, two-stage",
    "oracle": "Hindsight (upper bound)",
}
PARAM_LABEL = {"stock_ratio": "Stock / forecast", "salvage": "Salvage value", "dc_value": "DC stock value", "ship_cost_scale": "Shipping cost x"}


def load_results(root: Path) -> dict:
    res = {p.stem: read_json(p) for p in sorted((root / "results").glob("*.json"))}
    for need in ("data_summary", "launches", "forecast", "allocation"):
        if need not in res:
            raise FileNotFoundError(f"results/{need}.json is missing; run the pipeline first")
    return res


def pct(v: float, d: int = 1) -> str:
    return f"{100.0 * v:.{d}f}%"


def spct(v: float, d: int = 1) -> str:
    return f"{100.0 * v:+.{d}f}%"


def pp(v: float, d: int = 1) -> str:
    return f"{100.0 * v:+.{d}f} pp"


def num(v: float, d: int = 0) -> str:
    return f"{v:,.{d}f}"


def ci(pair, d: int = 3) -> str:
    return f"[{pair[0]:.{d}f}, {pair[1]:.{d}f}]"


def ci_share(pair, d: int = 1) -> str:
    return f"[{100.0 * pair[0]:.{d}f}%, {100.0 * pair[1]:.{d}f}%]"


def ci_pct(pair, d: int = 1) -> str:
    return f"[{100.0 * pair[0]:+.{d}f}%, {100.0 * pair[1]:+.{d}f}%]"


def ci_pp(pair, d: int = 1) -> str:
    return f"[{100.0 * pair[0]:+.{d}f}, {100.0 * pair[1]:+.{d}f}] pp"


def verdict(diff: dict, metric: str = "male") -> str:
    """Plain-language reading of a paired difference a - b on an error metric."""
    lo, hi = diff[f"{metric}_ci"]
    if hi < 0:
        return "better"
    if lo > 0:
        return "worse"
    return "not distinguishable"


def table(header: list[str], rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out)


# ------------------------------------------------------------------ facts


def facts(res: dict) -> dict:
    """Derived wording decisions shared by the README and the brief."""
    fc, al = res["forecast"], res["allocation"]
    cmp_ = fc["comparisons"]
    best = cmp_["best_simple_by_male"]
    pol = al["policies"]
    k_first = str(fc["params"]["post_launch_weeks"][0])
    k_last = str(fc["params"]["post_launch_weeks"][-1])
    sens = {(s["parameter"], s["value"]): s for s in al["sensitivity"]}
    return {
        "best_simple": best,
        "gbm_vs_best": verdict(cmp_["gradient_boosting_vs_best_simple_by_male"]),
        "gbm_vs_median": verdict(cmp_["gradient_boosting_vs_global_median"]),
        "best_vs_median": verdict(cmp_["analogues_vs_global_median"]),
        "calendar_vs_plain": verdict(cmp_["calendar_vs_no_calendar_boosting"]),
        "selected_vs_cv": verdict(cmp_["selected_vs_selected_random_cv"]),
        "k_first": k_first,
        "k_last": k_last,
        "bayes_first": verdict(fc["post_launch"][k_first]["bass_bayes_vs_best_simple"]),
        "bayes_last": verdict(fc["post_launch"][k_last]["bass_bayes_vs_best_simple"]),
        "bass_first": verdict(fc["post_launch"][k_first]["bass_vs_best_simple"]),
        "bass_last": verdict(fc["post_launch"][k_last]["bass_vs_best_simple"]),
        "nv_gain_sig": pol["newsvendor"]["value_gain_significant"],
        "two_gain_sig": pol["newsvendor_two_stage"]["value_gain_significant"],
        "dc_equal": sens.get(("dc_value", al["params"]["salvage"])),
        "sens": sens,
    }


# ----------------------------------------------------------------- README


def headline_rows(res: dict) -> list[list[str]]:
    ln, fc, al = res["launches"], res["forecast"], res["allocation"]
    f = facts(res)
    best = f["best_simple"]
    pre, pol = fc["pre_launch"], al["policies"]
    cov = fc["interval_coverage"]["selected"]
    k = f["k_first"]
    rows = [
        ["Launches identified / held out", f"{num(ln['funnel']['launches'])} / {num(fc['test_launches'])}",
         "`results/launches.json` funnel.launches; `results/forecast.json` test_launches"],
        [f"Best pre-launch method: {METHOD_LABEL[best].lower()}, MALE",
         f"{pre[best]['male']:.3f} {ci(pre[best]['male_ci'])} vs global median {pre['global_median']['male']:.3f}",
         f"`results/forecast.json` pre_launch.{best}.male"],
        ["Gradient boosting, MALE", f"{pre['gradient_boosting']['male']:.3f} {ci(pre['gradient_boosting']['male_ci'])} ({f['gbm_vs_best']} vs the best simple method)",
         "`results/forecast.json` pre_launch.gradient_boosting.male"],
        [f"{pct(cov['nominal'], 0)} prediction interval, coverage on held-out launches",
         f"{pct(cov['coverage'])} {ci_share(cov['wilson_ci'])}", "`results/forecast.json` interval_coverage.selected"],
        [f"After {k} weeks of sales: Bass + prior vs curve ratio, MALE of remaining demand",
         f"{fc['post_launch'][k]['methods']['bass_bayes']['male']:.3f} vs {fc['post_launch'][k]['methods']['curve_ratio']['male']:.3f}",
         f"`results/forecast.json` post_launch.{k}.methods"],
        ["MIP and marginal allocation agree (objective)", f"{num(al['mip_check']['objective_agree'])} of {num(al['mip_check']['launches'])} launches",
         "`results/allocation.json` mip_check"],
        ["Newsvendor vs proportional: value", f"{spct(pol['newsvendor']['value_gain_vs_proportional'], 2)} {ci_pct(pol['newsvendor']['value_gain_ci'], 2)}",
         "`results/allocation.json` policies.newsvendor.value_gain_vs_proportional"],
        ["Two-stage newsvendor vs proportional: value / fill rate / stock stranded in channels",
         f"{spct(pol['newsvendor_two_stage']['value_gain_vs_proportional'])} / {pp(pol['newsvendor_two_stage']['fill_rate_gain_vs_proportional'])} / {spct(pol['newsvendor_two_stage']['stranded_change_vs_proportional'])}",
         "`results/allocation.json` policies.newsvendor_two_stage"],
    ]
    return rows


def forecast_section(res: dict) -> str:
    fc = res["forecast"]
    f = facts(res)
    pre = fc["pre_launch"]
    cmp_ = fc["comparisons"]
    best = f["best_simple"]
    order = ["global_mean_curve", "global_median", "category_mean", "category_median", "nearest_analogue",
             f"analogues_k{fc['params']['analogue_k']}", "gradient_boosting", "gradient_boosting_calendar", "selected", "selected_random_cv"]
    rows = [[METHOD_LABEL[m], f"{pre[m]['male']:.3f} {ci(pre[m]['male_ci'])}", f"{pre[m]['wape']:.3f} {ci(pre[m]['wape_ci'])}", spct(pre[m]["bias"])] for m in order]
    d_gb, d_med, d_best = cmp_["gradient_boosting_vs_best_simple_by_male"], cmp_["gradient_boosting_vs_global_median"], cmp_["analogues_vs_global_median"]
    d_cal, d_sel = cmp_["calendar_vs_no_calendar_boosting"], cmp_["selected_vs_selected_random_cv"]
    sel = fc["selected_method_launches"]
    sel_cv = fc["selected_method_launches_random_cv"]
    lines = [
        f"Target: units sold in weeks 1-{fc['horizon_weeks']} after the first sale. {num(fc['test_launches'])} held-out launches "
        f"({num(fc['test_units'])} units) over {fc['origins_count']} monthly origins from {fc['first_origin']} to {fc['last_origin']}; "
        f"at each origin the models see only launches whose whole horizon had already ended (rule E1). "
        f"MALE = mean absolute error of log(1 + units); intervals are {pct(fc['params']['ci_level'], 0)} bootstrap intervals over launches "
        f"({num(fc['params']['bootstrap_reps'])} resamples).",
        "",
        table(["Method", "MALE", "WAPE", "Bias"], rows),
        "",
        "What the table says, plainly:",
        "",
        f"- The best simple method is the median of the {fc['params']['analogue_k']} nearest analogues. Against the global median its MALE differs by "
        f"{d_best['male']:+.3f} {ci(d_best['male_ci'])}: {f['best_vs_median']}.",
        f"- **Gradient boosting does not beat the simple methods.** Against the best simple method its MALE differs by {d_gb['male']:+.3f} {ci(d_gb['male_ci'])} "
        f"({f['gbm_vs_best']}); against the global median by {d_med['male']:+.3f} {ci(d_med['male_ci'])} ({f['gbm_vs_median']}).",
        f"- Adding calendar features (launch month, days to Christmas) to the boosting model changes MALE by {d_cal['male']:+.3f} {ci(d_cal['male_ci'])} ({f['calendar_vs_plain']}): "
        f"each training set spans less than one year, so the model extrapolates to months it has never seen.",
        f"- Choosing the method by random-fold cross-validation would have picked the calendar boosting model for "
        f"{num(sel_cv.get('gradient_boosting_calendar', 0))} of {num(fc['test_launches'])} launches and given MALE {pre['selected_random_cv']['male']:.3f}. "
        f"The forward-chaining rule E2 picks the analogue median for {num(sel.get(best, 0))} launches and gives {pre['selected']['male']:.3f} "
        f"(difference {d_sel['male']:+.3f} {ci(d_sel['male_ci'])}, {f['selected_vs_cv']}). The first version of this project used random folds; "
        f"the rule was changed after that result was seen on the held-out launches, which is disclosed under Limitations.",
        f"- Even the best method is rough: a MALE of {pre[best]['male']:.3f} is a typical error factor of about e^{pre[best]['male']:.3f}, and WAPE is {pct(pre[best]['wape'], 0)}.",
    ]
    return "\n".join(lines)


def interval_section(res: dict) -> str:
    fc = res["forecast"]
    cov = fc["interval_coverage"]
    order = ["selected", f"analogues_k{fc['params']['analogue_k']}", "global_median", "gradient_boosting", "gradient_boosting_quantile", "selected_random_cv"]
    label = dict(METHOD_LABEL, gradient_boosting_quantile="Gradient boosting, quantile loss (no calibration)")
    rows = []
    for m in order:
        c = cov[m]
        rows.append([label[m], f"{num(c['covered'])} / {num(c['n'])}", f"{pct(c['coverage'])} {ci_share(c['wilson_ci'])}",
                     "yes" if c["nominal_inside_ci"] else "no", f"{c['median_width_ratio']:.1f}"])
    nominal = pct(fc["params"]["pi_coverage"], 0)
    lines = [
        f"Nominal level {nominal}. Intervals are the point forecast shifted by split-conformal quantiles of forward-chaining validation residuals (rule F6). "
        f"Coverage is the share of held-out launches whose actual units fall inside, with a Wilson interval (rule M3). "
        f"Width is the median of (1 + upper) / (1 + lower).",
        "",
        table(["Interval", "Covered", "Coverage", f"{nominal} inside Wilson interval", "Median width ratio"], rows),
        "",
        f"The calibrated intervals reach the nominal level only by being very wide (upper bound about {cov['selected']['median_width_ratio']:.0f} times the lower bound). "
        f"Uncalibrated quantile boosting and intervals calibrated on random folds are much narrower and cover far less than {nominal}.",
    ]
    return "\n".join(lines)


def post_section(res: dict) -> str:
    fc = res["forecast"]
    f = facts(res)
    bs = fc["bass_shape"]
    out = [
        f"Target: units in the weeks after week k up to week {fc['horizon_weeks']}, given sales in weeks 1-k. Same held-out launches and origins.",
        "",
    ]
    header = ["Method"] + [f"k = {k}: MALE" for k in fc["params"]["post_launch_weeks"]] + [f"k = {k}: WAPE" for k in fc["params"]["post_launch_weeks"]]
    rows = []
    for m in ["prelaunch_only", "run_rate", "curve_ratio", "bass", "bass_bayes"]:
        r = [POST_LABEL[m]]
        for metric in ("male", "wape"):
            for k in fc["params"]["post_launch_weeks"]:
                b = fc["post_launch"][str(k)]["methods"][m]
                r.append(f"{b[metric]:.3f} {ci(b[metric + '_ci'])}")
        rows.append(r)
    out.append(table(header, rows))
    out.append("")
    kf, kl = f["k_first"], f["k_last"]
    df, dl = fc["post_launch"][kf]["bass_bayes_vs_best_simple"], fc["post_launch"][kl]["bass_bayes_vs_best_simple"]
    bf, bl = fc["post_launch"][kf]["bass_vs_best_simple"], fc["post_launch"][kl]["bass_vs_best_simple"]
    q_note = (
        f"The fitted imitation coefficient q sits at its lower bound ({bs['q_max']:.6f}) at every origin, with p between {bs['p_min']:.3f} and {bs['p_max']:.3f}: "
        "the average launch curve here is front-loaded with no later hump, so the Bass curve collapses to plain exponential saturation."
        if bs["q_at_lower_bound"] else
        f"Fitted Bass shape: p between {bs['p_min']:.3f} and {bs['p_max']:.3f}, q between {bs['q_min']:.3f} and {bs['q_max']:.3f}."
    )
    out += [
        f"- {q_note}",
        f"- Bass with the scale taken from sales alone vs the best simple update (curve ratio): MALE difference {bf['male']:+.3f} {ci(bf['male_ci'])} at k = {kf} ({f['bass_first']}) "
        f"and {bl['male']:+.3f} {ci(bl['male_ci'])} at k = {kl} ({f['bass_last']}).",
        f"- Bass with the pre-launch forecast as a prior: {df['male']:+.3f} {ci(df['male_ci'])} at k = {kf} ({f['bayes_first']}), "
        f"{dl['male']:+.3f} {ci(dl['male_ci'])} at k = {kl} ({f['bayes_last']}). The weight on sales is between {bs['weight_min']:.2f} and {bs['weight_max']:.2f}, "
        "so the prior only matters while very little has been sold.",
        "- Any update from real sales is far better than staying with the pre-launch forecast.",
    ]
    return "\n".join(out)


def allocation_section(res: dict) -> str:
    al = res["allocation"]
    f = facts(res)
    ap, pol, ctx, mc, rc = al["params"], al["policies"], al["context"], al["mip_check"], al["rule_cost"]
    names = al["channels"]
    ship = ", ".join(f"{n} {pct(ap['ship_cost'][n], 0)}" for n in names)
    mins = ", then ".join(f"{e['name']} {e['min_cases']}" for e in ap["priority_channels"])
    rows = []
    for p in ["proportional", "newsvendor_ship_all", "newsvendor", "proportional_two_stage", "newsvendor_two_stage", "oracle"]:
        b = pol[p]
        gain = "reference" if p == "proportional" else f"{spct(b['value_gain_vs_proportional'], 2)} {ci_pct(b['value_gain_ci'], 2)}"
        rows.append([POLICY_LABEL[p], pct(b["fill_rate"]), num(b["lost_units"]), pct(b["stranded_share_of_stock"]), num(b["leftover_dc_units"]),
                     num(b["value"]), gain, pct(b["value_share_of_oracle"])])
    nv, two, ship_all = pol["newsvendor"], pol["newsvendor_two_stage"], pol["newsvendor_ship_all"]
    share_rows = [[n, pct(ctx["demand_share"][n]), pct(ctx["mean_first_shipment_share"]["proportional"][n]), pct(ctx["mean_first_shipment_share"]["newsvendor"][n])] for n in names]
    sens_rows = []
    for s in al["sensitivity"]:
        sens_rows.append([PARAM_LABEL[s["parameter"]], f"{s['value']}", pct(s["proportional"]["fill_rate"]),
                          f"{spct(s['newsvendor_ship_all']['value_gain_vs_proportional'], 2)} {ci_pct(s['newsvendor_ship_all']['value_gain_ci'], 2)}",
                          f"{spct(s['newsvendor']['value_gain_vs_proportional'], 2)} {ci_pct(s['newsvendor']['value_gain_ci'], 2)}",
                          f"{spct(s['newsvendor_two_stage']['value_gain_vs_proportional'], 2)} {ci_pct(s['newsvendor_two_stage']['value_gain_ci'], 2)}"])
    dc_eq = f["dc_equal"]
    out = [
        f"Setting, per held-out launch: stock = {ap['stock_ratio']} x the pre-launch forecast, rounded up to whole cases (rule B1), split over {len(names)} channels (country groups). "
        f"Stock is already bought, so value = sales at the launch price + {pct(ap['salvage'], 0)} of price for units stranded in a channel "
        f"+ {pct(ap['dc_value'], 0)} of price for units still at the DC - shipping ({ship} of price). These cost figures are assumptions, not data.",
        "",
        f"Demand uncertainty: {ap['scenarios']} equally likely scenarios per launch, built from the forecast, its validation residuals and the channel shares of the "
        f"{ap['share_analogues']} nearest analogue launches (rule B2). Business rules, each an explicit constraint with its own unit tests: whole cases (A1); "
        f"minimum cases for priority channels, in this order: {mins} (A2); at most {pct(ap['max_share'], 0)} of the stock to one channel (A3).",
        "",
        f"The expected value is separable and concave in whole cases, so marginal (greedy) allocation is optimal. The same scenario problem solved as a MIP with HiGHS "
        f"gives the same objective on {num(mc['objective_agree'])} of {num(mc['launches'])} held-out launches (largest relative gap `{mc['max_relative_gap']:.1e}`; "
        f"the allocation itself is identical on {num(mc['identical_allocation'])}, the rest are ties). Unit tests add brute-force enumeration as a third check.",
        "",
        f"Back-test against the demand that actually occurred, replayed week by week (lost sales, no backorders). Across the {num(ctx['launches'])} launches "
        f"the stock is {num(ctx['stock_units'])} units against a demand of {num(ctx['demand_units'])}; stock is below demand for {num(ctx['launches_stock_below_demand'])} launches.",
        "",
        table(["Policy", "Fill rate", "Lost units", "Stranded in channels (share of stock)", "Left at DC (units)", "Value (GBP)", "Value vs proportional", "Share of hindsight value"], rows),
        "",
        "Reading it plainly:",
        "",
        f"- **Stock level, not the split, limits the fill rate.** Even with hindsight the same stock fills {pct(pol['oracle']['fill_rate'])} of demand, "
        f"because the forecast that sets the stock is rough. Proportional allocation already reaches {pct(pol['proportional']['fill_rate'])}.",
        f"- **The optimised one-shot split is a small gain.** Same stock, everything shipped: {spct(ship_all['value_gain_vs_proportional'], 2)} value {ci_pct(ship_all['value_gain_ci'], 2)}, "
        f"fill rate {pp(ship_all['fill_rate_gain_vs_proportional'])} {ci_pp(ship_all['fill_rate_gain_ci'])}. Allowed to keep cases at the DC: "
        f"{spct(nv['value_gain_vs_proportional'], 2)} value, fill rate {pp(nv['fill_rate_gain_vs_proportional'])}, stranded stock {spct(nv['stranded_change_vs_proportional'])}. "
        f"It is better on {num(nv['launches_better'])} launches and worse on {num(nv['launches_worse'])}.",
        f"- **Holding a reserve and re-allocating after week {ap['review_week']} is the larger lever, with a cost.** Keeping {pct(ap['reserve_fraction'], 0)} at the DC and shipping it after "
        f"the review gives {spct(two['value_gain_vs_proportional'])} value {ci_pct(two['value_gain_ci'])} and {spct(two['stranded_change_vs_proportional'])} stock stranded in channels, "
        f"but the fill rate moves by {pp(two['fill_rate_gain_vs_proportional'])} {ci_pp(two['fill_rate_gain_ci'])}: fast sellers run out before the review. "
        f"The proportional two-stage variant gains only {spct(pol['proportional_two_stage']['value_gain_vs_proportional'], 2)}.",
    ]
    if dc_eq is not None:
        out.append(
            f"- **That gain rests on one assumption.** When DC stock is valued no higher than stranded stock ({pct(dc_eq['value'], 0)} of price), the two-stage gain is "
            f"{spct(dc_eq['newsvendor_two_stage']['value_gain_vs_proportional'], 2)} {ci_pct(dc_eq['newsvendor_two_stage']['value_gain_ci'], 2)}."
        )
    out += [
        f"- The scenario model is optimistic: the newsvendor allocation realised {pct(nv['realised_over_expected'])} of the value its own scenarios promised "
        f"({pct(pol['proportional']['realised_over_expected'])} for the proportional allocation under the same scenarios).",
        "",
        "Average first shipment by channel (share of stock) next to the realised demand share:",
        "",
        table(["Channel", "Demand share", "Proportional", "Newsvendor"], share_rows),
        "",
        f"Cost of the business rules (newsvendor, realised value, relative to no A2/A3 rules): minimum only {spct(rc['minimum_only']['cost_vs_no_rules'], 2)} "
        f"(GBP {rc['minimum_only']['cost_per_launch']:.2f} per launch); maximum share only {spct(rc['max_share_only']['cost_vs_no_rules'], 2)}; both {spct(rc['both_rules']['cost_vs_no_rules'], 2)}. "
        f"A negative cost means the rule raised realised value in this back-test: with the cap the newsvendor allocation leaves {num(rc['max_share_only']['newsvendor_left_dc_units'])} units at the DC "
        f"against {num(rc['no_rules']['newsvendor_left_dc_units'])} without any rule, and DC stock is valued above stranded stock.",
        "",
        "Sensitivity (one parameter changed at a time; value gain vs proportional allocation):",
        "",
        table(["Parameter", "Value", "Proportional fill rate", "Newsvendor split, ship everything", "Newsvendor, may hold back", "Newsvendor, two-stage"], sens_rows),
    ]
    return "\n".join(out)


def verification_section(res: dict) -> str:
    mu = res.get("mutation")
    if not mu:
        return "Mutation testing has not been run yet (`python scripts/mutate.py`)."
    first, final = mu["first_run"], mu["final_run"]
    lines = [
        f"- {num(mu['tests_passed'])} offline tests (network access is blocked in `tests/conftest.py`), including hand-checked cases for every rule in `docs/rules.md`.",
        f"- Mutation testing (`scripts/mutate.py`, hand-written mutants of the rule code): the control mutant was killed first; "
        f"first run {num(first['killed'])} of {num(first['mutants'])} mutants killed ({num(first['survived'])} survived), "
        f"final run {num(final['killed'])} of {num(final['mutants'])} after adding tests for the survivors. Both runs are recorded in `results/mutation.json`.",
        "- `README.md`, `docs/rules.md` and `docs/decision_brief.md` are generated by `scripts/make_docs.py`; tests regenerate them, compare them with the committed files and trace every printed number to `results/*.json`.",
    ]
    if final["survived"]:
        lines.append(f"- {num(final['survived'])} mutants still survive; they are listed with reasons in `results/mutation.json`.")
    return "\n".join(lines)


def render_readme(res: dict) -> str:
    ds, ln, fc, al = res["data_summary"], res["launches"], res["forecast"], res["allocation"]
    f = facts(res)
    fn, cl = ln["funnel"], ds["cleaning"]
    best = f["best_simple"]
    pre, pol = fc["pre_launch"], al["policies"]
    nv, two = pol["newsvendor"], pol["newsvendor_two_stage"]
    cov = fc["interval_coverage"]["selected"]
    kf = f["k_first"]
    names = al["channels"]
    parts = [
        FIRST_LINE,
        "",
        "# New-product forecasting and channel allocation",
        "",
        "Two decisions a product team takes for every launch, written as explicit, testable rules and checked on real data: "
        "how much will a new product sell in its first weeks, and how should a limited first batch be split across sales channels? "
        "Every rule has a parameter file entry, a unit test and a result it affects (`docs/rules.md`); a short Chinese and English decision brief is in `docs/decision_brief.md`.",
        "",
        "## Headline results",
        "",
        table(["Result", "Value", "Source"], headline_rows(res)),
        "",
        "## Data and licence",
        "",
        f"- {ds['source']['dataset']}, UCI Machine Learning Repository: invoice lines of a UK online giftware retailer, {ds['first_line']} to {ds['last_line']}. "
        f"Licence: {ds['source']['licence']}. Citation: {ds['source']['citation']}",
        f"- Retrieved {ds['source']['retrieved']} from the UCI archive (`scripts/fetch_data.py`); `SHA-256` of the archive `{ds['sha256']['online_retail_ii.zip']}`. "
        "The raw file is not in this repository; `data/derived/` holds small aggregated extracts (no customer identifiers).",
        f"- {num(cl['raw_rows'])} raw rows; {num(cl['overlap_rows_dropped'])} duplicated by the overlap of the two workbook sheets, {num(cl['cancellation_rows'])} cancellation rows, "
        f"{num(cl['non_product_rows'])} non-product rows, {num(cl['orders_reversed_by_cancellation'])} order lines reversed by a later cancellation; {num(cl['demand_lines'])} demand lines remain.",
        f"- Channels are {ds['channels']} country groups ({ds['countries']} countries): " + ", ".join(f"{n} {pct(ds['channel_unit_share'][n])}" for n in names) + " of units.",
        "",
        "## A. Launch identification",
        "",
        f"A launch is a product code first sold on or after {ln['params']['burn_in_end']} (the first months are burn-in), with its whole {ln['params']['horizon_weeks']}-week horizon inside the data, "
        "that passes three re-code checks.",
        "",
        table(["Step", "Products"], [
            ["Product codes with demand", num(fn["products"])],
            ["First sold during burn-in (existing range)", num(fn["first_sale_in_burn_in"])],
            ["First sold after burn-in", num(fn["first_sale_after_burn_in"])],
            ["Horizon not fully observed", num(fn["horizon_not_observed"])],
            ["Candidates", num(fn["candidates"])],
            ["Rejected: new variant of an older base code (L3)", num(fn["rejected_variant"])],
            ["Rejected: same description as an older code (L4)", num(fn["rejected_same_description"])],
            ["Rejected: successor of a phased-out code (L5)", num(fn["rejected_succession"])],
            ["Launches", num(fn["launches"])],
        ]),
        "",
        f"Launch curves are weekly units per channel for weeks 1-{ln['params']['horizon_weeks']}. Median launch: {num(ln['units_12w']['median'])} units "
        f"(10th to 90th percentile {num(ln['units_12w']['p10'])} to {num(ln['units_12w']['p90'])}), median launch price GBP {ln['median_launch_price']:.2f}. "
        f"The average curve is front-loaded: week 1 holds {pct(fc['launch_curve']['mean_weekly_share'][0])} of a launch's {ln['params']['horizon_weeks']}-week units "
        f"and the first {fc['params']['post_launch_weeks'][-1]} weeks {pct(fc['launch_curve']['mean_cumulative_share'][fc['params']['post_launch_weeks'][-1] - 1])}.",
        "",
        "## A. Pre-launch forecast",
        "",
        forecast_section(res),
        "",
        "### Prediction intervals",
        "",
        interval_section(res),
        "",
        "### Update after the first weeks of sales",
        "",
        post_section(res),
        "",
        "## B. Channel allocation of a limited first batch",
        "",
        allocation_section(res),
        "",
        "## C. Rules and decision brief",
        "",
        f"- `docs/rules.md`: {len(RULES)} rules, each with its parameters (from `config/params.json`), the tests that verify it and the result keys it affects.",
        "- `docs/decision_brief.md`: one page in Chinese and English, generated from the results.",
        "- `docs/sources.md`: every external fact with URL and retrieval date, the data licence and a list of what could not be verified.",
        "",
        "## Verification",
        "",
        verification_section(res),
        "",
        "## Reproduce",
        "",
        "```",
        "python -m venv .venv && .venv/Scripts/python -m pip install -r requirements.txt   # Windows; use .venv/bin/python elsewhere",
        "python -m pytest                      # offline; uses data/derived and results only",
        "python scripts/run_pipeline.py        # re-runs forecasts and the allocation back-test from data/derived",
        "python scripts/make_docs.py           # regenerates README.md, docs/rules.md, docs/decision_brief.md",
        "python scripts/mutate.py              # mutation testing",
        "python -I scripts/fetch_data.py && python -I scripts/prepare_data.py   # optional: rebuild data/derived from the raw file",
        "```",
        "",
        "## Limitations",
        "",
        "- The data are one giftware retailer whose customers are largely wholesalers; countries stand in for sales channels and one of them holds most of the demand. "
        "Nothing here is evidence about any other company or product category.",
        "- Recorded sales are treated as demand. Stock-outs at the retailer are not visible in the data, so true demand may have been higher.",
        "- Prices, salvage value, DC value and shipping costs in the allocation model are assumptions; the sensitivity table shows how much the conclusions move with them, "
        "and the two-stage gain mostly disappears when DC stock is not worth more than stranded stock.",
        "- The case-pack size is a proxy (the most frequent order quantity inside the horizon); it is used for rounding only, never as a forecasting feature.",
        "- The launch price is read from the invoices of week 1 as a stand-in for a list price that would be known before launch; it is a forecasting feature.",
        "- A product first sold after the burn-in may still be a relaunch from before the data window; the three re-code checks cannot see that. The successor check (L5) found no case, which may also mean its thresholds are too strict.",
        "- Model selection was changed once after seeing held-out results: random-fold cross-validation chose a model that failed out of time, so the rule now uses forward-chaining validation, "
        "and calendar features were removed from the default boosting model. Both earlier choices are kept in the results as comparisons. The held-out launches are therefore not untouched by design decisions.",
        "- Forward-chaining validation inside the training set has no gap between its own training and validation launches, unlike the real origin; interval calibration may be slightly optimistic.",
        "- The two-stage review forecasts remaining demand from recorded sales, which are censored when a channel has sold out; it does not correct for that.",
        "- The first-stage shipment of the two-stage policy is a heuristic (one-shot marginal allocation on the non-reserved stock), not an optimal two-stage stochastic programme.",
        "- The GitHub Actions workflow has not been run: this repository is local only. The reproduction test of the full pipeline is marked slow and is not part of the default run.",
        "",
        "## 中文摘要",
        "",
        f"个人项目，使用公开数据（UCI Online Retail II，{ds['source']['licence'].split(' (')[0]}），与任何公司无关。把“新品预测”和“渠道分货”两个决策写成可量化、可验证的规则：",
        "",
        f"- 新品识别：在 {num(fn['products'])} 个商品编码中，经预热期与三项换码检查，识别出 {num(fn['launches'])} 个新品，其中 {num(fc['test_launches'])} 个按上市月份滚动留出做评测，训练只用已走完 {fc['horizon_weeks']} 周的新品。",
        f"- 上市前预测：相似品（文本+价格最近的 {fc['params']['analogue_k']} 个）中位数最好，对数误差 MALE {pre[best]['male']:.3f}，全局中位数为 {pre['global_median']['male']:.3f}；"
        f"梯度提升为 {pre['gradient_boosting']['male']:.3f}，没有胜过简单基线。{pct(cov['nominal'], 0)} 预测区间在留出集上的实际覆盖率为 {pct(cov['coverage'])}，但区间很宽。",
        f"- 上市后更新：有 {kf} 周销量后，带先验的 Bass 更新 MALE 为 {fc['post_launch'][kf]['methods']['bass_bayes']['male']:.3f}，曲线比例法为 {fc['post_launch'][kf]['methods']['curve_ratio']['male']:.3f}。",
        f"- 渠道分货：按期望边际价值分配（报童模型），与 MIP 精确解在 {num(al['mip_check']['objective_agree'])} / {num(al['mip_check']['launches'])} 个新品上目标值一致；"
        f"相对“按历史占比分货”，一次性分货价值 {spct(nv['value_gain_vs_proportional'], 2)}；留 {pct(al['params']['reserve_fraction'], 0)} 库存、第 {al['params']['review_week']} 周后补货的两阶段策略价值 {spct(two['value_gain_vs_proportional'])}、"
        f"渠道滞留库存 {spct(two['stranded_change_vs_proportional'])}，但满足率 {pp(two['fill_rate_gain_vs_proportional'])}。",
        "- 局限：成本参数为假设值；销量被当作需求；单一零售商数据，国家分组充当渠道；模型选择规则在看到留出结果后改过一次（已在英文局限中说明）。",
        "",
        "## Licence",
        "",
        "Code: MIT, see `LICENSE`. Data: see above; the derived extracts in `data/derived/` are adapted from the UCI dataset under its licence.",
        "",
    ]
    return "\n".join(parts)


# ---------------------------------------------------------------- rules.md


def lookup(obj, dotted: str):
    cur = obj
    for part in dotted.split("."):
        cur = cur[part]
    return cur


def fmt_param(path: str, params: dict) -> str:
    v = lookup(params, path)
    if isinstance(v, (dict, list)):
        if isinstance(v, dict) and all(not isinstance(x, (dict, list)) for x in v.values()):
            return f"`{path}` = " + ", ".join(f"{k} {x}" for k, x in v.items())
        if isinstance(v, list) and v and all(isinstance(x, dict) and set(x) == {"name", "min_cases"} for x in v):
            return f"`{path}` = " + ", then ".join(f"{x['name']} {x['min_cases']}" for x in v)
        return f"`{path}` (list in `config/params.json`)"
    return f"`{path}` = {v}"


def render_rules(params: dict) -> str:
    out = [
        "# Decision rules",
        "",
        "Generated by `scripts/make_docs.py` from `src/npalloc/rules_registry.py` and `config/params.json`. "
        "`tests/test_docs.py` checks that every parameter exists, every test exists and every result key exists.",
        "",
    ]
    groups: list[str] = []
    for r in RULES:
        if r["group"] not in groups:
            groups.append(r["group"])
    for g in groups:
        out += [f"## {g}", ""]
        rows = []
        for r in RULES:
            if r["group"] != g:
                continue
            ps = "<br>".join(fmt_param(p, params) for p in r["params"]) or "none"
            ts = "<br>".join(f"`tests/{t}`" for t in r["tests"])
            af = "<br>".join(f"`results/{a.split(':')[0]}` `{a.split(':')[1]}`" for a in r["affects"])
            rows.append([r["id"], r["statement"], ps, ts, af])
        out.append(table(["Rule", "Statement", "Parameters", "Verified by", "Result it affects"], rows))
        out.append("")
    return "\n".join(out)


# ------------------------------------------------------------------- brief


def render_brief(res: dict) -> str:
    ln, fc, al = res["launches"], res["forecast"], res["allocation"]
    f = facts(res)
    best = f["best_simple"]
    pre, pol, ap = fc["pre_launch"], al["policies"], al["params"]
    nv, two, prop = pol["newsvendor"], pol["newsvendor_two_stage"], pol["proportional"]
    cov = fc["interval_coverage"]["selected"]
    kf = f["k_first"]
    post = fc["post_launch"][kf]["methods"]
    dc_eq = f["dc_equal"]
    en = [
        "# Decision brief",
        "",
        "Generated by `scripts/make_docs.py`; every number is traced to `results/*.json` by `tests/test_docs.py`. " + FIRST_LINE,
        "",
        "## English",
        "",
        f"**Question.** For a new product, how many units will sell in the first {fc['horizon_weeks']} weeks, and how should the first batch be split across {len(al['channels'])} channels?",
        "",
        f"**Evidence.** {num(ln['funnel']['launches'])} launches identified in public retail data; {num(fc['test_launches'])} of them are forecast out of time, "
        f"each one only from launches whose {fc['horizon_weeks']} weeks had already ended before its launch month.",
        "",
        "**Recommendations.**",
        "",
        f"- Forecast before launch with the median of the {fc['params']['analogue_k']} most similar earlier launches (description text and price). MALE {pre[best]['male']:.3f} "
        f"against {pre['global_median']['male']:.3f} for the global median and {pre['gradient_boosting']['male']:.3f} for gradient boosting: the complex model is not worth deploying here.",
        f"- Quote the forecast as a range. The {pct(cov['nominal'], 0)} interval covered {pct(cov['coverage'])} of held-out launches, and its upper bound is about {cov['median_width_ratio']:.0f} times its lower bound, "
        "so a single number hides most of the risk.",
        f"- Re-forecast as soon as {kf} weeks of sales exist: MALE of remaining demand falls from {post['prelaunch_only']['male']:.3f} (no update) to {post['bass_bayes']['male']:.3f} (Bass curve with the pre-launch forecast as prior); "
        f"the plain curve ratio gives {post['curve_ratio']['male']:.3f}.",
        f"- Split the first batch by expected marginal value rather than by historical share: {spct(nv['value_gain_vs_proportional'], 2)} value {ci_pct(nv['value_gain_ci'], 2)} with the same stock and the same business rules. "
        f"The gain is small: one channel holds {pct(al['context']['demand_share'][al['channels'][0]])} of demand, so there is little to re-split.",
        f"- Keep {pct(ap['reserve_fraction'], 0)} of the batch at the DC and release it after week {ap['review_week']}: {spct(two['value_gain_vs_proportional'])} value and {spct(two['stranded_change_vs_proportional'])} stock stranded in channels, "
        f"at the price of a fill rate of {pct(two['fill_rate'])} instead of {pct(prop['fill_rate'])}."
        + (f" If DC stock is worth no more than stranded stock the value gain is only {spct(dc_eq['newsvendor_two_stage']['value_gain_vs_proportional'], 2)}." if dc_eq else ""),
        "",
        f"**What limits the result.** With stock set from the forecast, even a hindsight split fills {pct(pol['oracle']['fill_rate'])} of demand: the forecast, not the split, is the bottleneck. "
        "Cost parameters are assumptions; recorded sales stand in for demand.",
        "",
        "## 中文",
        "",
        f"**问题。** 新品上市后前 {fc['horizon_weeks']} 周能卖多少？首批库存如何在 {len(al['channels'])} 个渠道之间分配？",
        "",
        f"**依据。** 在公开零售数据中识别出 {num(ln['funnel']['launches'])} 个新品，其中 {num(fc['test_launches'])} 个按上市月份滚动留出：预测每个新品时，只用其上市月之前已走完 {fc['horizon_weeks']} 周的新品。",
        "",
        "**建议。**",
        "",
        f"- 上市前用最相似的 {fc['params']['analogue_k']} 个历史新品（品名文本+价格）的中位数做预测：MALE {pre[best]['male']:.3f}，全局中位数 {pre['global_median']['male']:.3f}，梯度提升 {pre['gradient_boosting']['male']:.3f}；复杂模型在这里不值得上线。",
        f"- 预测要给区间：{pct(cov['nominal'], 0)} 区间在留出新品上的覆盖率为 {pct(cov['coverage'])}，上界约为下界的 {cov['median_width_ratio']:.0f} 倍，只报一个数会掩盖大部分风险。",
        f"- 有了 {kf} 周销量就重新预测：剩余需求的 MALE 从 {post['prelaunch_only']['male']:.3f}（不更新）降到 {post['bass_bayes']['male']:.3f}（带先验的 Bass 曲线），曲线比例法为 {post['curve_ratio']['male']:.3f}。",
        f"- 首批分货按期望边际价值而不是按历史占比：同样库存、同样业务规则下价值 {spct(nv['value_gain_vs_proportional'], 2)}（区间 {ci_pct(nv['value_gain_ci'], 2)}）；单一渠道占需求的 {pct(al['context']['demand_share'][al['channels'][0]])}，可重新分配的空间小，增益不大。",
        f"- 留 {pct(ap['reserve_fraction'], 0)} 在仓、第 {ap['review_week']} 周后按实际销量补货：价值 {spct(two['value_gain_vs_proportional'])}，渠道滞留库存 {spct(two['stranded_change_vs_proportional'])}，代价是满足率从 {pct(prop['fill_rate'])} 变为 {pct(two['fill_rate'])}。"
        + (f"若在仓库存并不比滞留库存更值钱，价值增益只有 {spct(dc_eq['newsvendor_two_stage']['value_gain_vs_proportional'], 2)}。" if dc_eq else ""),
        "",
        f"**限制。** 库存按预测设定时，即使事后最优的分法也只能满足 {pct(pol['oracle']['fill_rate'])} 的需求：瓶颈在预测，不在分货。成本参数为假设；销量被当作需求。",
        "",
    ]
    return "\n".join(en)


def render_all(root: Path, params: dict) -> dict[str, str]:
    res = load_results(root)
    return {
        "README.md": render_readme(res),
        "docs/rules.md": render_rules(params),
        "docs/decision_brief.md": render_brief(res),
    }
