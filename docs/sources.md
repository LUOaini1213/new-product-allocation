# Sources

Every external fact used in this repository, where it comes from and when it was retrieved.
Everything else in the documents is computed from the data by the code in `src/npalloc/`.

## Data

| Fact | Source | Source date | Retrieved |
|---|---|---|---|
| Dataset name "Online Retail II"; creator Daqing Chen; 1,067,371 instances; "a real online retail transaction data set of two years" | https://archive.ics.uci.edu/dataset/502/online+retail+ii | donated 2019-09-20 | 2026-11-15 |
| The data contain all transactions of a UK-based, registered, non-store online retailer between 01/12/2009 and 09/12/2011; the company mainly sells all-occasion gift-ware; many customers are wholesalers | same page, dataset description | donated 2019-09-20 | 2026-11-15 |
| Variable notes: an invoice number starting with "c" is a cancellation; StockCode is a five-digit product code; the unit price is in pounds sterling; Country is where the customer resides | same page, "Additional Variable Information" | donated 2019-09-20 | 2026-11-15 |
| Licence of the dataset: Creative Commons Attribution 4.0 International (CC BY 4.0) | same page, licence field | donated 2019-09-20 | 2026-11-15 |
| Citation: Chen, D. (2012). Online Retail II [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5CG6D | same page, citation field | 2012 (as printed) | 2026-11-15 |
| Download location of the archive | https://archive.ics.uci.edu/static/public/502/online+retail+ii.zip | not stated | 2026-11-15 |
| CC BY 4.0 allows copying, redistribution and adaptation, including commercially, provided credit is given, the licence is linked and changes are indicated | https://creativecommons.org/licenses/by/4.0/ | not stated | 2026-11-17 |

### Files as retrieved

| File | Bytes | SHA-256 |
|---|---|---|
| `online_retail_ii.zip` (archive) | 45,622,418 | `572e36277c2390fbfde10664750731e0a86f55e33470d91919085f0408e67bfb` |
| `online_retail_II.xlsx` (the only member of the archive) | 45,622,278 | `bcbe73b35f5b7babf197fb0cb983a11f5d9ff929078d4aa53d171b1f2df2e980` |

The same values are in `data/hashes.json`; `scripts/fetch_data.py` refuses to continue when they do not match.

### Licence and attribution of the derived files

The raw files are not redistributed here. The files in `data/derived/` and `results/per_launch.csv` are adapted from the
dataset (cleaned, aggregated to product x week x country group; no invoice numbers or customer identifiers) and are shared
under the same CC BY 4.0 licence with the citation above. The changes made are the cleaning and launch rules in `docs/rules.md`.

## Software

| Fact | Source | Retrieved |
|---|---|---|
| `scipy.optimize.milp` is a wrapper of the HiGHS solver | docstring of `scipy.optimize.milp` in the pinned SciPy version (`requirements.txt`) | 2026-11-17 |
| Package versions | `requirements.txt` (output of `pip freeze` in the project environment) | 2026-11-15 |

## Could not verify

- Whether the recorded sales were ever limited by the retailer's own stock. The dataset has no inventory data; sales are treated as demand.
- Unit costs, margins, salvage values and shipping costs. None are in the data; the values in `config/params.json` are assumptions.
- Whether a product first sold after the burn-in period existed before the data window starts (a relaunch would look like a launch).
- The meaning of a letter suffix on a stock code. The UCI page only says the code is a five-digit number; treating suffixes as variants of one base product is an inference from the descriptions.
- Why the two workbook sheets overlap in time. The UCI page does not mention it; the overlapping rows were checked to be identical before one copy was dropped.
- The UCI page shows no value for the number of features and an empty variable table; the eight column names used here are the workbook's own headers.
- Literature attributions were written from memory and not checked against the original publications: the diffusion model of Bass (1969, Management Science; the publisher's page refused automated access on 2026-11-17), split-conformal prediction intervals, the Wilson score interval, the Hamilton (largest-remainder) method, and the optimality of greedy marginal allocation for separable concave objectives. The formulas actually used are stated in `docs/rules.md` and in the module docstrings, and are checked numerically by the tests (the greedy optimum against a MIP and against brute-force enumeration).
- The GitHub Actions workflow in `.github/workflows/tests.yml` has never been executed; this repository has no remote.
