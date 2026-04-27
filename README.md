# Credibilistic Portfolio Optimisation — CTFN + MOGA

Python replication of:

> Mandal, P.K., Thakur, M., & Mittal, G. (2024).  
> *Credibilistic portfolio optimization with higher-order moments using coherent triangular fuzzy numbers.*  
> Applied Soft Computing, 151, 111155.  
> https://doi.org/10.1016/j.asoc.2023.111155

---

## What this replicates

Three multi-objective portfolio optimisation models using **Coherent Triangular Fuzzy Numbers (CTFN)** in the credibilistic framework:

| Model | Objectives |
|-------|-----------|
| **Model I** | Maximise Mean, Minimise Semivariance, Maximise Skewness, Minimise Semikurtosis |
| **Model II** | Maximise Mean, Minimise MASD, Maximise Skewness, Minimise Semikurtosis |
| **Model III** | Maximise Mean, Minimise CVaR, Maximise Skewness, Minimise Semikurtosis |

Applied to two markets:
- **NSE** (NIFTY 50 constituents, 2014–2022)
- **NYSE** (DJIA constituents, 2014–2022)

---

## Notebooks

| Notebook | Purpose |
|----------|---------|
| `credibilistic_portfolio.ipynb` | Data pipeline — downloads prices, computes returns, selects 18 stocks per market, splits train/test |
| `credibilistic_moga.ipynb` | Main implementation — CTFN moment formulas (Eqs. 3–8), MOGA operators, NSGA-II selection, K-medoids clustering, results |

---

## Key implementation details

- **CTFN fitting** (Section 4.1): percentile-based (Q3, Q20, Q50, Q80, Q97) with random shape parameter `k`
- **Credibilistic moments**: closed-form Eqs. 3–8 (mean, semivariance, MASD, CVaR, skewness, semikurtosis)
- **MOGA operators**: CCBEX crossover, swap mutation, power mutation, repair mechanism (Appendices B.1–B.4)
- **Selection**: NSGA-II (non-dominated sorting + crowding distance)
- **Representative solutions**: K-medoids clustering (KMeans-medoid approximation, k=25)
- **Filtration**: mean ≥ 2% monthly AND skewness ≥ 0

---

## Data

`data/` contains pre-processed monthly return series (Jul 2014 – Jun 2022):

```
data/
├── nse_train.csv        # NSE training returns  (72 months × 18 stocks)
├── nse_test.csv         # NSE test returns      (24 months × 18 stocks)
├── nyse_train.csv       # NYSE training returns (72 months × 18 stocks)
├── nyse_test.csv        # NYSE test returns     (24 months × 18 stocks)
├── nifty_stock_key.json # S1–S18 → ticker mapping (NSE)
└── djia_stock_key.json  # S1–S18 → ticker mapping (NYSE)
```

Returns are stored as **decimal fractions** (e.g. 0.025 = 2.5% monthly return).

---

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install numpy pandas scikit-learn matplotlib yfinance jupyter
```

---

## Running

**Quick test** (~2.5 min, verifies pipeline):
```bash
jupyter nbconvert --to notebook --execute credibilistic_moga.ipynb \
  --output credibilistic_moga_out.ipynb --ExecutePreprocessor.timeout=1800
```

**Full replication** (~4–8 hrs, matches paper Tables 6–11):  
Set `QUICK_TEST = False` in cell `c009`, then:
```bash
nohup jupyter nbconvert --to notebook --execute credibilistic_moga.ipynb \
  --output credibilistic_moga_full.ipynb \
  --ExecutePreprocessor.timeout=36000 > moga_run.log 2>&1 &
```

---

## Constraints (Table 3 of paper)

| Parameter | Value |
|-----------|-------|
| Assets per market (n) | 18 |
| Cardinality (κ) | 5 |
| Lower bound (l) | 0.08 |
| Upper bound (u) | 0.30 |
| CVaR confidence (p) | 0.95 |
| Population size | 180 |
| Generations | 2000 |
| Independent runs | 30 |
| Representative solutions | 25 |

---

## Formula fixes vs paper

Three bugs were identified and fixed during replication:

1. **Mean (Eq. 3)**: Paper uses `b2 + (β − kα) / (2(k+1))` — an earlier reading mistakenly parsed it as `b2 + β/2 − kα/(2(k+1))`
2. **Semikurtosis Case 1 coefficient (Eq. 8)**: Correct coefficient is `12`, not `0.5`
3. **Semikurtosis Case 2 last term (Eq. 8)**: Correct coefficient is `12`, not `0.5`

Both semikurtosis coefficients follow directly from the Beta function `B(1/k, 5) = 24k⁵·c′₄`, giving `(1/2k)·24k⁵·c′₄ = 12k⁴·c′₄`.
