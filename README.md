Personal project on public data; not affiliated with any company.

# New-product forecasting and channel allocation

Two decisions a product team takes for every launch, written as explicit, testable rules and checked on real data: how much will a new product sell in its first weeks, and how should a limited first batch be split across sales channels? Every rule has a parameter file entry, a unit test and a result it affects (`docs/rules.md`); a short Chinese and English decision brief is in `docs/decision_brief.md`.

## Headline results

| Result | Value | Source |
|---|---|---|
| Launches identified / held out | 822 / 437 | `results/launches.json` funnel.launches; `results/forecast.json` test_launches |
| Best pre-launch method: median of nearest analogues, MALE | 1.147 [1.053, 1.245] vs global median 1.239 | `results/forecast.json` pre_launch.analogues_k10.male |
| Gradient boosting, MALE | 1.235 [1.136, 1.334] (worse vs the best simple method) | `results/forecast.json` pre_launch.gradient_boosting.male |
| 80% prediction interval, coverage on held-out launches | 82.6% [78.8%, 85.9%] | `results/forecast.json` interval_coverage.selected |
| After 2 weeks of sales: Bass + prior vs curve ratio, MALE of remaining demand | 0.744 vs 0.800 | `results/forecast.json` post_launch.2.methods |
| MIP and marginal allocation agree (objective) | 437 of 437 launches | `results/allocation.json` mip_check |
| Newsvendor vs proportional: value | +0.95% [+0.61%, +1.30%] | `results/allocation.json` policies.newsvendor.value_gain_vs_proportional |
| Two-stage newsvendor vs proportional: value / fill rate / stock stranded in channels | +5.3% / -2.5 pp / -40.3% | `results/allocation.json` policies.newsvendor_two_stage |

## Data and licence

- Online Retail II, UCI Machine Learning Repository: invoice lines of a UK online giftware retailer, 2009-12-01 to 2011-12-09. Licence: CC BY 4.0 (Creative Commons Attribution 4.0 International). Citation: Chen, D. (2012). Online Retail II [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5CG6D
- Retrieved 2026-11-15 from the UCI archive (`scripts/fetch_data.py`); `SHA-256` of the archive `572e36277c2390fbfde10664750731e0a86f55e33470d91919085f0408e67bfb`. The raw file is not in this repository; `data/derived/` holds small aggregated extracts (no customer identifiers).
- 1,067,371 raw rows; 22,523 duplicated by the overlap of the two workbook sheets, 19,165 cancellation rows, 5,992 non-product rows, 4,442 order lines reversed by a later cancellation; 1,010,503 demand lines remain.
- Channels are 7 country groups (43 countries): United Kingdom 81.8%, EIRE 3.0%, Germany 2.0%, France 2.5%, Netherlands 3.5%, Other Europe 5.5%, Rest of World 1.7% of units.

## A. Launch identification

A launch is a product code first sold on or after 2010-06-01 (the first months are burn-in), with its whole 12-week horizon inside the data, that passes three re-code checks.

| Step | Products |
|---|---|
| Product codes with demand | 4,700 |
| First sold during burn-in (existing range) | 3,651 |
| First sold after burn-in | 1,049 |
| Horizon not fully observed | 191 |
| Candidates | 858 |
| Rejected: new variant of an older base code (L3) | 34 |
| Rejected: same description as an older code (L4) | 2 |
| Rejected: successor of a phased-out code (L5) | 0 |
| Launches | 822 |

Launch curves are weekly units per channel for weeks 1-12. Median launch: 468 units (10th to 90th percentile 35 to 2,160), median launch price GBP 2.08. The average curve is front-loaded: week 1 holds 17.5% of a launch's 12-week units and the first 4 weeks 43.5%.

## A. Pre-launch forecast

Target: units sold in weeks 1-12 after the first sale. 437 held-out launches (384,535 units) over 9 monthly origins from 2011-01-01 to 2011-09-01; at each origin the models see only launches whose whole horizon had already ended (rule E1). MALE = mean absolute error of log(1 + units); intervals are 95% bootstrap intervals over launches (2,000 resamples).

| Method | MALE | WAPE | Bias |
|---|---|---|---|
| Global mean launch curve | 1.343 [1.237, 1.458] | 0.888 [0.827, 0.962] | +2.4% |
| Global median | 1.239 [1.152, 1.333] | 0.802 [0.772, 0.834] | -47.3% |
| Category mean curve | 1.384 [1.276, 1.502] | 1.005 [0.917, 1.104] | +4.2% |
| Category median | 1.337 [1.241, 1.438] | 0.866 [0.821, 0.915] | -40.5% |
| Nearest analogue (text + price) | 1.437 [1.328, 1.557] | 1.274 [1.100, 1.466] | +49.3% |
| Median of nearest analogues | 1.147 [1.053, 1.245] | 0.754 [0.702, 0.809] | -19.1% |
| Gradient boosting (attributes) | 1.235 [1.136, 1.334] | 0.781 [0.728, 0.837] | -35.3% |
| Gradient boosting + calendar features | 1.674 [1.560, 1.797] | 0.858 [0.817, 0.899] | -64.9% |
| Selected per origin (rule E2) | 1.175 [1.082, 1.272] | 0.757 [0.706, 0.810] | -21.9% |
| Selected by random-fold CV (comparison) | 1.525 [1.413, 1.636] | 0.825 [0.782, 0.874] | -50.2% |

What the table says, plainly:

- The best simple method is the median of the 10 nearest analogues. Against the global median its MALE differs by -0.092 [-0.157, -0.025]: better.
- **Gradient boosting does not beat the simple methods.** Against the best simple method its MALE differs by +0.087 [0.010, 0.168] (worse); against the global median by -0.005 [-0.098, 0.083] (not distinguishable).
- Adding calendar features (launch month, days to Christmas) to the boosting model changes MALE by +0.440 [0.336, 0.546] (worse): each training set spans less than one year, so the model extrapolates to months it has never seen.
- Choosing the method by random-fold cross-validation would have picked the calendar boosting model for 305 of 437 launches and given MALE 1.525. The forward-chaining rule E2 picks the analogue median for 408 launches and gives 1.175 (difference -0.350 [-0.462, -0.240], better). The first version of this project used random folds; the rule was changed after that result was seen on the held-out launches, which is disclosed under Limitations.
- Even the best method is rough: a MALE of 1.147 is a typical error factor of about e^1.147, and WAPE is 75%.

### Prediction intervals

Nominal level 80%. Intervals are the point forecast shifted by split-conformal quantiles of forward-chaining validation residuals (rule F6). Coverage is the share of held-out launches whose actual units fall inside, with a Wilson interval (rule M3). Width is the median of (1 + upper) / (1 + lower).

| Interval | Covered | Coverage | 80% inside Wilson interval | Median width ratio |
|---|---|---|---|---|
| Selected per origin (rule E2) | 361 / 437 | 82.6% [78.8%, 85.9%] | yes | 64.1 |
| Median of nearest analogues | 365 / 437 | 83.5% [79.8%, 86.7%] | yes | 64.1 |
| Global median | 330 / 437 | 75.5% [71.3%, 79.3%] | no | 43.7 |
| Gradient boosting (attributes) | 373 / 437 | 85.4% [81.7%, 88.4%] | no | 93.9 |
| Gradient boosting, quantile loss (no calibration) | 238 / 437 | 54.5% [49.8%, 59.1%] | no | 7.2 |
| Selected by random-fold CV (comparison) | 250 / 437 | 57.2% [52.5%, 61.8%] | no | 24.2 |

The calibrated intervals reach the nominal level only by being very wide (upper bound about 64 times the lower bound). Uncalibrated quantile boosting and intervals calibrated on random folds are much narrower and cover far less than 80%.

### Update after the first weeks of sales

Target: units in the weeks after week k up to week 12, given sales in weeks 1-k. Same held-out launches and origins.

| Method | k = 2: MALE | k = 4: MALE | k = 2: WAPE | k = 4: WAPE |
|---|---|---|---|---|
| Pre-launch forecast minus sales (no update) | 2.015 [1.817, 2.212] | 2.554 [2.336, 2.782] | 0.935 [0.869, 1.004] | 1.082 [1.006, 1.170] |
| Run rate | 0.845 [0.771, 0.922] | 0.730 [0.666, 0.797] | 0.811 [0.708, 0.939] | 0.586 [0.511, 0.672] |
| Curve ratio | 0.800 [0.731, 0.874] | 0.668 [0.608, 0.733] | 0.537 [0.471, 0.619] | 0.437 [0.389, 0.491] |
| Bass, scale from sales | 0.806 [0.738, 0.880] | 0.681 [0.621, 0.745] | 0.511 [0.451, 0.586] | 0.438 [0.395, 0.487] |
| Bass, scale from sales + prior | 0.744 [0.675, 0.821] | 0.661 [0.603, 0.724] | 0.466 [0.424, 0.511] | 0.462 [0.428, 0.498] |

- The fitted imitation coefficient q sits at its lower bound (0.000001) at every origin, with p between 0.051 and 0.086: the average launch curve here is front-loaded with no later hump, so the Bass curve collapses to plain exponential saturation.
- Bass with the scale taken from sales alone vs the best simple update (curve ratio): MALE difference +0.006 [-0.002, 0.014] at k = 2 (not distinguishable) and +0.012 [0.003, 0.021] at k = 4 (worse).
- Bass with the pre-launch forecast as a prior: -0.055 [-0.090, -0.021] at k = 2 (better), -0.007 [-0.031, 0.015] at k = 4 (not distinguishable). The weight on sales is between 0.72 and 0.86, so the prior only matters while very little has been sold.
- Any update from real sales is far better than staying with the pre-launch forecast.

## B. Channel allocation of a limited first batch

Setting, per held-out launch: stock = 1.0 x the pre-launch forecast, rounded up to whole cases (rule B1), split over 7 channels (country groups). Stock is already bought, so value = sales at the launch price + 30% of price for units stranded in a channel + 50% of price for units still at the DC - shipping (United Kingdom 2%, EIRE 6%, Germany 8%, France 8%, Netherlands 8%, Other Europe 10%, Rest of World 15% of price). These cost figures are assumptions, not data.

Demand uncertainty: 200 equally likely scenarios per launch, built from the forecast, its validation residuals and the channel shares of the 20 nearest analogue launches (rule B2). Business rules, each an explicit constraint with its own unit tests: whole cases (A1); minimum cases for priority channels, in this order: United Kingdom 1, then EIRE 1 (A2); at most 90% of the stock to one channel (A3).

The expected value is separable and concave in whole cases, so marginal (greedy) allocation is optimal. The same scenario problem solved as a MIP with HiGHS gives the same objective on 437 of 437 held-out launches (largest relative gap `6.3e-16`; the allocation itself is identical on 402, the rest are ties). Unit tests add brute-force enumeration as a third check.

Back-test against the demand that actually occurred, replayed week by week (lost sales, no backorders). Across the 437 launches the stock is 301,880 units against a demand of 384,535; stock is below demand for 197 launches.

| Policy | Fill rate | Lost units | Stranded in channels (share of stock) | Left at DC (units) | Value (GBP) | Value vs proportional | Share of hindsight value |
|---|---|---|---|---|---|---|---|
| Proportional to history | 46.9% | 204,162 | 40.3% | 0 | 429,119 | reference | 83.0% |
| Newsvendor split, ship everything | 47.1% | 203,369 | 40.0% | 0 | 432,287 | +0.74% [+0.41%, +1.06%] | 83.6% |
| Newsvendor, may hold back | 46.4% | 206,118 | 38.2% | 8,100 | 433,179 | +0.95% [+0.61%, +1.30%] | 83.8% |
| Proportional, two-stage | 47.2% | 202,988 | 39.9% | 0 | 430,796 | +0.39% [+0.18%, +0.61%] | 83.3% |
| Newsvendor, two-stage | 44.4% | 213,613 | 24.0% | 58,399 | 451,842 | +5.30% [+4.43%, +6.25%] | 87.4% |
| Hindsight (upper bound) | 51.0% | 188,300 | 0.6% | 103,726 | 517,063 | +20.49% [+18.50%, +22.69%] | 100.0% |

Reading it plainly:

- **Stock level, not the split, limits the fill rate.** Even with hindsight the same stock fills 51.0% of demand, because the forecast that sets the stock is rough. Proportional allocation already reaches 46.9%.
- **The optimised one-shot split is a small gain.** Same stock, everything shipped: +0.74% value [+0.41%, +1.06%], fill rate +0.2 pp [-0.1, +0.5] pp. Allowed to keep cases at the DC: +0.95% value, fill rate -0.5 pp, stranded stock -5.1%. It is better on 299 launches and worse on 136.
- **Holding a reserve and re-allocating after week 4 is the larger lever, with a cost.** Keeping 30% at the DC and shipping it after the review gives +5.3% value [+4.4%, +6.2%] and -40.3% stock stranded in channels, but the fill rate moves by -2.5 pp [-3.2, -1.9] pp: fast sellers run out before the review. The proportional two-stage variant gains only +0.39%.
- **That gain rests on one assumption.** When DC stock is valued no higher than stranded stock (30% of price), the two-stage gain is +0.25% [-0.20%, +0.69%].
- The scenario model is optimistic: the newsvendor allocation realised 91.8% of the value its own scenarios promised (92.5% for the proportional allocation under the same scenarios).

Average first shipment by channel (share of stock) next to the realised demand share:

| Channel | Demand share | Proportional | Newsvendor |
|---|---|---|---|
| United Kingdom | 77.3% | 82.3% | 88.9% |
| EIRE | 3.0% | 3.3% | 3.0% |
| Germany | 3.0% | 2.0% | 1.1% |
| France | 2.3% | 2.9% | 0.8% |
| Netherlands | 5.1% | 3.6% | 1.4% |
| Other Europe | 4.3% | 4.1% | 2.1% |
| Rest of World | 4.9% | 1.8% | 0.2% |

Cost of the business rules (newsvendor, realised value, relative to no A2/A3 rules): minimum only +0.02% (GBP 0.17 per launch); maximum share only -0.17%; both -0.19%. A negative cost means the rule raised realised value in this back-test: with the cap the newsvendor allocation leaves 8,915 units at the DC against 1,174 without any rule, and DC stock is valued above stranded stock.

Sensitivity (one parameter changed at a time; value gain vs proportional allocation):

| Parameter | Value | Proportional fill rate | Newsvendor split, ship everything | Newsvendor, may hold back | Newsvendor, two-stage |
|---|---|---|---|---|---|
| Stock / forecast | 0.5 | 28.4% | +1.84% [+1.52%, +2.18%] | +1.84% [+1.52%, +2.18%] | +3.29% [+2.62%, +4.00%] |
| Stock / forecast | 1.5 | 58.6% | +0.38% [+0.07%, +0.69%] | +2.75% [+1.92%, +3.60%] | +8.12% [+7.27%, +8.98%] |
| Salvage value | 0.1 | 46.9% | +0.85% [+0.37%, +1.32%] | +7.04% [+4.95%, +9.26%] | +13.16% [+11.13%, +15.39%] |
| Salvage value | 0.5 | 46.9% | +0.66% [+0.45%, +0.87%] | +0.66% [+0.45%, +0.87%] | +0.34% [+0.04%, +0.63%] |
| DC stock value | 0.3 | 46.9% | +0.74% [+0.41%, +1.06%] | +0.74% [+0.41%, +1.06%] | +0.25% [-0.20%, +0.69%] |
| Shipping cost x | 0.5 | 46.9% | +0.33% [+0.03%, +0.63%] | +0.40% [+0.10%, +0.69%] | +4.52% [+3.72%, +5.35%] |
| Shipping cost x | 2.0 | 46.9% | +1.66% [+1.31%, +2.02%] | +2.61% [+2.15%, +3.09%] | +7.42% [+6.38%, +8.57%] |

## C. Rules and decision brief

- `docs/rules.md`: 37 rules, each with its parameters (from `config/params.json`), the tests that verify it and the result keys it affects.
- `docs/decision_brief.md`: one page in Chinese and English, generated from the results.
- `docs/sources.md`: every external fact with URL and retrieval date, the data licence and a list of what could not be verified.

## Verification

- 131 offline tests (network access is blocked in `tests/conftest.py`), including hand-checked cases for every rule in `docs/rules.md`.
- Mutation testing (`scripts/mutate.py`, hand-written mutants of the rule code): the control mutant was killed first; first run 76 of 86 mutants killed (10 survived), final run 86 of 86 after adding tests for the survivors. Both runs are recorded in `results/mutation.json`.
- `README.md`, `docs/rules.md` and `docs/decision_brief.md` are generated by `scripts/make_docs.py`; tests regenerate them, compare them with the committed files and trace every printed number to `results/*.json`.

## Reproduce

```
python -m venv .venv && .venv/Scripts/python -m pip install -r requirements.txt   # Windows; use .venv/bin/python elsewhere
python -m pytest                      # offline; uses data/derived and results only
python scripts/run_pipeline.py        # re-runs forecasts and the allocation back-test from data/derived
python scripts/make_docs.py           # regenerates README.md, docs/rules.md, docs/decision_brief.md
python scripts/mutate.py              # mutation testing
python -I scripts/fetch_data.py && python -I scripts/prepare_data.py   # optional: rebuild data/derived from the raw file
```

## Limitations

- The data are one giftware retailer whose customers are largely wholesalers; countries stand in for sales channels and one of them holds most of the demand. Nothing here is evidence about any other company or product category.
- Recorded sales are treated as demand. Stock-outs at the retailer are not visible in the data, so true demand may have been higher.
- Prices, salvage value, DC value and shipping costs in the allocation model are assumptions; the sensitivity table shows how much the conclusions move with them, and the two-stage gain mostly disappears when DC stock is not worth more than stranded stock.
- The case-pack size is a proxy (the most frequent order quantity inside the horizon); it is used for rounding only, never as a forecasting feature.
- The launch price is read from the invoices of week 1 as a stand-in for a list price that would be known before launch; it is a forecasting feature.
- A product first sold after the burn-in may still be a relaunch from before the data window; the three re-code checks cannot see that. The successor check (L5) found no case, which may also mean its thresholds are too strict.
- Model selection was changed once after seeing held-out results: random-fold cross-validation chose a model that failed out of time, so the rule now uses forward-chaining validation, and calendar features were removed from the default boosting model. Both earlier choices are kept in the results as comparisons. The held-out launches are therefore not untouched by design decisions.
- Forward-chaining validation inside the training set has no gap between its own training and validation launches, unlike the real origin; interval calibration may be slightly optimistic.
- The two-stage review forecasts remaining demand from recorded sales, which are censored when a channel has sold out; it does not correct for that.
- The first-stage shipment of the two-stage policy is a heuristic (one-shot marginal allocation on the non-reserved stock), not an optimal two-stage stochastic programme.
- The GitHub Actions workflow has not been run: this repository is local only. The reproduction test of the full pipeline is marked slow and is not part of the default run.

## 中文摘要

个人项目，使用公开数据（UCI Online Retail II，CC BY 4.0），与任何公司无关。把“新品预测”和“渠道分货”两个决策写成可量化、可验证的规则：

- 新品识别：在 4,700 个商品编码中，经预热期与三项换码检查，识别出 822 个新品，其中 437 个按上市月份滚动留出做评测，训练只用已走完 12 周的新品。
- 上市前预测：相似品（文本+价格最近的 10 个）中位数最好，对数误差 MALE 1.147，全局中位数为 1.239；梯度提升为 1.235，没有胜过简单基线。80% 预测区间在留出集上的实际覆盖率为 82.6%，但区间很宽。
- 上市后更新：有 2 周销量后，带先验的 Bass 更新 MALE 为 0.744，曲线比例法为 0.800。
- 渠道分货：按期望边际价值分配（报童模型），与 MIP 精确解在 437 / 437 个新品上目标值一致；相对“按历史占比分货”，一次性分货价值 +0.95%；留 30% 库存、第 4 周后补货的两阶段策略价值 +5.3%、渠道滞留库存 -40.3%，但满足率 -2.5 pp。
- 局限：成本参数为假设值；销量被当作需求；单一零售商数据，国家分组充当渠道；模型选择规则在看到留出结果后改过一次（已在英文局限中说明）。

## Licence

Code: MIT, see `LICENSE`. Data: see above; the derived extracts in `data/derived/` are adapted from the UCI dataset under its licence.
