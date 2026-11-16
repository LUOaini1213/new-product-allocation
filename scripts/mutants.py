"""Hand-written mutants of the rule code, used by scripts/mutate.py.

Each entry: (id, file under src/npalloc, exact text to replace, replacement, rule it attacks).
The text to replace must occur exactly once in the file.
"""
from __future__ import annotations

CONTROL = (
    "control", "allocation.py",
    "    return float(per.sum() - ship.sum() + dc)",
    "    return 0.0",
    "control: expected value always zero (must be killed, otherwise the harness is broken)",
)

MUTANTS = [
    # ---------------------------------------------------------------- cleaning
    ("c1_three_letters", "cleaning.py", '"^[0-9]{5}[A-Z]{0,2}$"', '"^[0-9]{5}[A-Z]{0,3}$"', "C1"),
    ("c1_four_digits", "cleaning.py", '"^[0-9]{5}[A-Z]{0,2}$"', '"^[0-9]{4,5}[A-Z]{0,2}$"', "C1"),
    ("c2_prefix", "cleaning.py", '.startswith("C")', '.startswith("A")', "C2"),
    ("c3_same_timestamp", "cleaning.py", "            if t0 > ts:\n                continue", "            if t0 >= ts:\n                continue", "C3"),
    ("c3_window_edge", "cleaning.py", "if t0 < ts - window:", "if t0 <= ts - window:", "C3"),
    ("c3_quantity_sign", "cleaning.py", "key = (cust, code, int(-qty))", "key = (cust, code, int(qty))", "C3"),
    ("c4_zero_price", "cleaning.py", '(std["qty"] > 0) & (std["price"] > 0)', '(std["qty"] > 0) & (std["price"] >= 0)', "C4"),
    ("c0_overlap_edge", "cleaning.py", '(raw["sheet"] == 0) & (raw["InvoiceDate"] >= start1)', '(raw["sheet"] == 0) & (raw["InvoiceDate"] > start1)', "C0"),
    # ---------------------------------------------------------------- launches
    ("l1_burn_in_edge", "launches.py", 'prod["first_day"] >= burn_in_end', 'prod["first_day"] > burn_in_end', "L1"),
    ("l2_horizon_edge", "launches.py", "<= data_end_excl", "< data_end_excl", "L2"),
    ("l3_colaunch_edge", "launches.py", ".dt.days > colaunch_days", ".dt.days >= colaunch_days", "L3"),
    ("l4_not_strictly_earlier", "launches.py", 'named["first_day"] > earliest_desc', 'named["first_day"] >= earliest_desc', "L4"),
    ("l5_jaccard_edge", "launches.py", "best_j >= jaccard_min", "best_j > jaccard_min", "L5"),
    ("l5_no_upper_gap", "launches.py", "(last >= lc - gap) & (last <= lc + gap)", "(last >= lc - gap)", "L5"),
    ("l5_price_band", "launches.py", "<= max_log + 1e-12", "<= 2 * max_log + 1e-12", "L5"),
    ("l6_week_offset", "launches.py", ".dt.days // 7 + 1", ".dt.days // 7", "L6"),
    ("l6_mode_tie", "launches.py", "return sorted(top)[0]", "return sorted(top)[-1]", "L6"),
    ("l6_price_mean", "launches.py", 'w1.groupby("code")["price"].median()', 'w1.groupby("code")["price"].mean()', "L6"),
    ("jaccard_denominator", "textutil.py", "return len(a & b) / len(a | b)", "return len(a & b) / max(len(a), len(b))", "L5"),
    ("ch1_fallback", "channels.py", 'return params["channels"]["fallback"]', 'return params["channels"]["groups"][0]["name"]', "CH1"),
    ("g1_fallback", "categories.py", 'return params["categories"]["fallback"]', 'return params["categories"]["rules"][0]["name"]', "G1"),
    # ---------------------------------------------------------------- forecast
    ("f1_median_to_mean", "forecast.py", "self.v = float(np.median(train.total))", "self.v = float(np.mean(train.total))", "F1"),
    ("f2_min_count_edge", "forecast.py", "if len(grp) >= self.min_count:", "if len(grp) > self.min_count:", "F2"),
    ("f4_price_sign", "forecast.py",
     "return (1.0 - cos) + self.price_weight * np.abs(logp[:, None] - self.logp[None, :])",
     "return (1.0 - cos) - self.price_weight * np.abs(logp[:, None] - self.logp[None, :])", "F4"),
    ("f4_median_to_mean", "forecast.py", "return np.median(self.y[nn], axis=1)", "return np.mean(self.y[nn], axis=1)", "F4"),
    ("f4_skip_nearest", "forecast.py", 'kind="stable")[:, :k]', 'kind="stable")[:, 1:k + 1]', "F4"),
    ("f4_self_is_neighbour", "forecast.py", "np.fill_diagonal(d, np.inf)", "np.fill_diagonal(d, 0.0)", "F4"),
    ("f5_calendar_flag", "forecast.py", "if not self.calendar:", "if self.calendar:", "F5"),
    ("f5_xmas_edge", "forecast.py", "if day > xmas:", "if day >= xmas:", "F5"),
    ("f6_one_sided", "forecast.py", "a = (1.0 - coverage) / 2.0", "a = (1.0 - coverage)", "F6"),
    ("f6_low_rank", "forecast.py", "lo = r[max(lo_rank, 1) - 1]", "lo = r[max(lo_rank, 1)]", "F6"),
    ("f6_upper_transform", "forecast.py", "np.maximum(np.expm1(lp + q_lo), 0.0), np.expm1(lp + q_hi)", "np.maximum(np.expm1(lp + q_lo), 0.0), np.exp(lp + q_hi)", "F6"),
    ("e2_forward_leak", "forecast.py", "(np.concatenate(chunks[:b]), chunks[b])", "(np.concatenate(chunks[:b + 1]), chunks[b])", "E2"),
    ("e2_forward_start", "forecast.py", "for b in range(min_train_blocks, blocks)]", "for b in range(min_train_blocks - 1, blocks)]", "E2"),
    ("cv_fold_leak", "forecast.py", "m = method.fit(train.take(~te))", "m = method.fit(train)", "E2"),
    # -------------------------------------------------------------------- bass
    ("p3_cdf_ratio", "bass.py", "return (1.0 - e) / (1.0 + (q / p) * e)", "return (1.0 - e) / (1.0 + (p / q) * e)", "P3"),
    ("p1_weeks_left", "bass.py", "/ k * (horizon - k)", "/ k * horizon", "P1"),
    ("p2_median_to_mean", "bass.py", "return float(np.median(rest[ok] / first[ok]))", "return float(np.mean(rest[ok] / first[ok]))", "P2"),
    ("p3_scale", "bass.py", "return np.asarray(sold_k, dtype=float) * (fh - fk) / fk", "return np.asarray(sold_k, dtype=float) * (fh - fk) / fh", "P3"),
    ("p4_weight_swapped", "bass.py", "return float(var_prior / (var_prior + var_data))", "return float(var_data / (var_prior + var_data))", "P4"),
    ("p4_no_floor", "bass.py", "m_data = np.maximum(np.asarray(sold_k, dtype=float), 1.0) / fk", "m_data = np.maximum(np.asarray(sold_k, dtype=float), 1e-12) / fk", "P4"),
    # ----------------------------------------------------------------- metrics
    ("m1_wape_denominator", "metrics.py", "return float(np.abs(f - a).sum() / a.sum())", "return float(np.abs(f - a).sum() / f.sum())", "M1"),
    ("m1_bias_sign", "metrics.py", "return float((f - a).sum() / a.sum())", "return float((a - f).sum() / a.sum())", "M1"),
    ("m2_one_sided", "metrics.py", "    a = (1.0 - level) / 2.0\n    return float(np.quantile(vals, a))", "    a = (1.0 - level)\n    return float(np.quantile(vals, a))", "M2"),
    ("m3_wilson_centre", "metrics.py", "centre = (ph + z * z / (2 * n)) / den", "centre = (ph + z * z / (4 * n)) / den", "M3"),
    # -------------------------------------------------------------- allocation
    ("a4_strict_exceedance", "allocation.py", 'np.searchsorted(d, start[c] + ks, side="left")', 'np.searchsorted(d, start[c] + ks, side="right")', "A4"),
    ("a4_ship_cost_sign", "allocation.py", "econ.price * ((v - econ.ship_cost)[:, None] + (1.0 - v) * prob_ge)", "econ.price * ((v + econ.ship_cost)[:, None] + (1.0 - v) * prob_ge)", "A4"),
    ("a1_pack_sum", "allocation.py", "reshape(n_ch, n_cases, pack).sum(axis=2)", "reshape(n_ch, n_cases, pack).max(axis=2)", "A1"),
    ("a2_minimum_ignores_stock", "allocation.py", "give = min(int(rules.min_cases[c]), left)", "give = int(rules.min_cases[c])", "A2"),
    ("a3_no_single_case", "allocation.py", "cap = max(1, int(np.floor(", "cap = max(0, int(np.floor(", "A3"),
    ("a3_minimum_does_not_override", "allocation.py", "    upper = np.maximum(upper, lower)\n    return lower, upper", "    return lower, upper", "A3"),
    ("a4_keep_tie", "allocation.py", "if not ship_all and -neg <= keep:", "if not ship_all and -neg < keep:", "A4"),
    ("a4_budget_ignores_minimums", "allocation.py", "    budget -= int(cases.sum())", "    budget -= 0", "A4"),
    ("a4_keep_value_per_unit", "allocation.py", "keep = econ.price * econ.dc_value * pack", "keep = econ.price * econ.dc_value", "A4"),
    ("a4_mip_stock", "allocation.py", "LinearConstraint(a2, -np.inf, float(q_units)),", "LinearConstraint(a2, -np.inf, float(q_units) + pack),", "A4"),
    ("a5_remainder_tie", "allocation.py", "key=lambda i: (-rem[i], i))", "key=lambda i: (-rem[i], -i))", "A5"),
    ("a5_donor_tie", "allocation.py", "donor = int(np.flatnonzero(slack == slack.max())[-1])", "donor = int(np.flatnonzero(slack == slack.max())[0])", "A5"),
    ("a5_destination_tie", "allocation.py", "dest = int(np.flatnonzero(gap == gap.max())[0])", "dest = int(np.flatnonzero(gap == gap.max())[-1])", "A5"),
    # ---------------------------------------------------------------- simulate
    ("s1_sell_without_stock", "simulate.py", "s = np.minimum(onhand, demand[w])", "s = demand[w].copy()", "S1"),
    ("s1_review_week", "simulate.py", "and w + 1 == review_week:", "and w == review_week:", "S1"),
    ("s1_shipping_sign", "simulate.py", "- p * float(np.dot(econ.ship_cost, shipped))", "+ p * float(np.dot(econ.ship_cost, shipped))", "B4"),
    ("s1_negative_review", "simulate.py", "if (extra < 0).any() or extra.sum() > dc:", "if extra.sum() > dc:", "S1"),
    ("s1_lost_sales", "simulate.py", "lost += demand[w] - s", "lost += demand[w]", "S1"),
    ("s1_dc_value_dropped", "simulate.py", "        + econ.dc_value * p * left_dc\n", "", "B4"),
    # ---------------------------------------------------------------- pipeline
    ("e1_horizon_edge", "pipeline.py", "<= origin).to_numpy()", "< origin).to_numpy()", "E1"),
    ("e1_horizon_leak", "pipeline.py", 'train = (day + pd.Timedelta(days=7 * horizon_weeks) <= origin).to_numpy()', "train = (day <= origin).to_numpy()", "E1"),
    ("e1_test_month_end", "pipeline.py", "(day < origin + pd.DateOffset(months=1))", "(day <= origin + pd.DateOffset(months=1))", "E1"),
    ("e2_picks_worst", "pipeline.py", "if best is None or v < cv_male[best]:", "if best is None or v > cv_male[best]:", "E2"),
    ("a5_history_window", "pipeline.py", "sel = cm[(months >= lo) & (months < origin)]", "sel = cm[(months < origin)]", "A5"),
    ("a5_history_includes_origin", "pipeline.py", "sel = cm[(months >= lo) & (months < origin)]", "sel = cm[(months >= lo) & (months <= origin)]", "A5"),
    ("p2_ratio_inverted", "pipeline.py", "return sold_k * self.ratio", "return sold_k / self.ratio", "P2"),
    ("post_selection_on_all_rows", "pipeline.py", "rem, sold_v = (total - sold)[rows], sold[rows]", "rem, sold_v = (total - sold)[: len(rows)], sold[: len(rows)]", "P2"),
    # ---------------------------------------------------------------- backtest
    ("b1_floor", "backtest.py", "cases = int(np.ceil(ratio * forecast / pack - 1e-9))", "cases = int(np.floor(ratio * forecast / pack - 1e-9))", "B1"),
    ("b1_zero_stock", "backtest.py", "return max(cases, 1) * pack", "return max(cases, 0) * pack", "B1"),
    ("b2_prior_weight", "backtest.py", "return (u + prior_units * prior) / (u.sum(axis=-1, keepdims=True) + prior_units)", "return (u + prior) / (u.sum(axis=-1, keepdims=True) + 1.0)", "B2"),
    ("b2_quantile_grid", "backtest.py", "(np.arange(n) + 0.5) / n", "np.arange(n) / n", "B2"),
    ("b2_floor_units", "backtest.py", "return np.rint(totals[:, None] * share_rows[pick]).astype(np.int64)", "return np.floor(totals[:, None] * share_rows[pick]).astype(np.int64)", "B2"),
    ("b3_reserve_swapped", "backtest.py", "np.ceil((1.0 - reserve_fraction) * cases - 1e-9)", "np.ceil(reserve_fraction * cases - 1e-9)", "B3"),
    ("b4_fill_rate_denominator", "backtest.py", '"fill_rate": float(a["sales"].sum() / a["demand"].sum()),', '"fill_rate": float(a["sales"].sum() / stock.sum()),', "B4"),
    ("b4_stranded_includes_dc", "backtest.py", '"stranded_share_of_stock": float(a["left_channel"].sum() / stock.sum()),', '"stranded_share_of_stock": float((a["left_channel"].sum() + a["left_dc"].sum()) / stock.sum()),', "B4"),
    ("a3_rule_switch", "backtest.py", 'max_share=float(ap["max_share"]) if use_max else 1.0', 'max_share=float(ap["max_share"]) if not use_max else 1.0', "A3"),
    ("a2_priority_order_dropped", "backtest.py", '            order.append(names.index(entry["name"]))\n', "", "A2"),
    ("b4_gain_sign", "backtest.py", 'dv = a["value"] - ref["value"]', 'dv = ref["value"] - a["value"]', "B4"),
    # ---------------------------------------------------------------- numcheck
    ("numcheck_percent_always", "numcheck.py", "if pct and abs(round(100.0 * a, decimals) - value) <= tol:", "if abs(round(100.0 * a, decimals) - value) <= tol:", "docs"),
    ("numcheck_skips_dates", "numcheck.py", "        if d not in blob:\n            bad.append(d)", "        if d not in blob:\n            pass", "docs"),
    ("numcheck_loose_rounding", "numcheck.py", "if abs(round(a, decimals) - value) <= tol:", "if abs(round(a, decimals) - value) <= 0.05:", "docs"),
]
