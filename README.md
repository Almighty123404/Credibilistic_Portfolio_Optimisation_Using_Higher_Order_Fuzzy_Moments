# Credibilistic Portfolio Optimisation Using Higher-Order Fuzzy Moments

A full Python replication of:

> **Mandal, P.K., Thakur, M., & Mittal, G. (2024)**  
> *Credibilistic portfolio optimization with higher-order moments using coherent triangular fuzzy numbers.*  
> Applied Soft Computing, 151, 111155.  
> https://doi.org/10.1016/j.asoc.2023.111155

---

## What This Project Does

Classical portfolio optimisation (Markowitz) only uses mean and variance. This paper — and this replication — goes further by incorporating **four higher-order moments** of the portfolio return distribution, modelled using **Coherent Triangular Fuzzy Numbers (CTFN)** in the **credibilistic framework**:

| Moment | Why it matters |
|--------|---------------|
| **Mean** | Expected return — maximise |
| **Semivariance / MASD / CVaR** | Downside risk — minimise |
| **Skewness** | Prefer right-skewed returns (big upside) — maximise |
| **Semikurtosis** | Penalise heavy left tail (catastrophic losses) — minimise |

Three models are formulated, each using a different risk measure:

| Model | Risk measure |
|-------|-------------|
| Model I | Semivariance (SV) |
| Model II | Mean Absolute Semi-Deviation (MASD) |
| Model III | Conditional Value-at-Risk (CVaR) |

All three are solved as **4-objective optimisation problems** using a **Multi-Objective Genetic Algorithm (MOGA)** adapted with cardinality constraints, and tested on two real stock markets.

---

## Datasets

Two markets, both using **monthly returns from July 2014 to June 2022** (96 months total):

| Market | Index | Candidate pool | Final selection |
|--------|-------|---------------|-----------------|
| **NSE** (India) | NIFTY 50 | All constituents during 2014–2022 | 18 stocks |
| **NYSE** (USA) | DJIA | All constituents during 2014–2022 | 18 stocks |

**Train/test split:** 72 months training (Jul 2014 – Jun 2020), 24 months test (Jul 2020 – Jun 2022).

### NSE stocks selected (18)
Adani Ports, SBI, JSW Steel, Larsen & Toubro, Lupin, Mahindra & Mahindra, Maruti Suzuki, Nestle India, NTPC, ONGC, Power Grid, Reliance Industries, Shree Cement, Indian Oil, Sun Pharma, Tata Consumer, Tata Steel, TCS

### NYSE stocks selected (18)
Apple, Procter & Gamble, McDonald's, 3M, Merck, Microsoft, Nike, Pfizer, Raytheon Technologies, Amgen, AT&T, Travelers, UnitedHealth, Visa, Verizon, Walmart, Coca-Cola, JPMorgan Chase

### Stock selection methodology
Starting from all index members present at any point during 2014–2022 (~37 DJIA / ~52 NIFTY candidates), stocks are filtered by:
1. **Data completeness**: must have full 96-month price history
2. **Tiebreaker** (when >18 pass): lowest mean absolute monthly return during the training window

---

## Theory — How the Fuzzy Model Works

### Step 1: Portfolio return time series
For a portfolio **x** with κ=5 active assets:
$$R_t(\mathbf{x}) = \sum_{i=1}^{n} r_{ti} \cdot x_i \cdot z_i, \quad t = 1, \ldots, T$$

where $r_{ti}$ is the decimal monthly return of asset $i$ at time $t$, and $z_i \in \{0,1\}$ is the activity indicator.

### Step 2: Fit a CTFN to the portfolio returns
A **Coherent Triangular Fuzzy Number** $\tilde{A} = (b_1, b_2, b_3)_k$ is fitted to the time series $\{R_t\}$ using percentiles (Section 4.1 of the paper):

| Parameter | Formula |
|-----------|---------|
| $b_1$ | $\min(R_{\min},\ Q_3)$ — left anchor |
| $b_2$ | $Q_{50}$ — median (mode) |
| $b_3$ | $Q_{97}$ — right anchor |
| $k$ | Shape parameter, computed from $Q_{20}$ or $Q_{80}$ using $k = \frac{\ln(0.5)}{\ln(\text{denom})}$ |

The shape parameter $k$ controls tail asymmetry — larger $k$ means heavier left tail.

### Step 3: Compute the four credibilistic moments (closed-form)

The **credibility measure** is the average of possibility and necessity — a self-dual measure that avoids the inconsistency of pure possibility theory. All moments have closed-form expressions derived from the CTFN membership function.

**CTFN membership function** (with $\alpha = b_2 - b_1$, $\beta = b_3 - b_2$):

- For $b_1 \leq t \leq b_2$: &nbsp; $\mu_{\tilde{A}}(t) = \left(\dfrac{t-b_1}{\alpha}\right)^k$
- For $b_2 \leq t \leq b_3$: &nbsp; $\mu_{\tilde{A}}(t) = \left(\dfrac{b_3-t}{\beta}\right)^k$

**Constants used across all formulas:**

$$c_j = \frac{1}{(k+1)(k+2)\cdots(k+j)} \qquad c'_j = \frac{1}{(1+k)(1+2k)\cdots(1+jk)}$$

---

**Eq. 3 — Credibilistic Mean:**

$$E[\tilde{A}] = b_2 + \frac{\beta - k\alpha}{2(k+1)}$$

---

**Eq. 4 — Credibilistic Semivariance** — two cases on the sign of $e - b_2$:

- If $e \leq b_2$: &nbsp; $\text{SV}[\tilde{A}] = k^2\, \rho_1^{1/k+2}\, c'_2 \,/\, \alpha^{1/k}$
- If $e > b_2$: &nbsp; $\text{SV}[\tilde{A}] = \Xi_1 - \rho_3^{k+2}\, c_2 \,/\, \beta^k$

---

**Eq. 5 — Credibilistic MASD** — two cases on whether $k\alpha \geq \beta$:

If $k\alpha \geq \beta$:

$$\text{MASD}[\tilde{A}] = \frac{k\alpha}{2(k+1)}\left(1 + \frac{\beta - k\alpha}{2\alpha(k+1)}\right)^{(k+1)/k}$$

If $k\alpha < \beta$:

$$\text{MASD}[\tilde{A}] = \frac{\beta}{2(k+1)}\left(1 + \frac{k\alpha - \beta}{2\beta(k+1)}\right)^{k+1}$$

---

**Eq. 6 — Credibilistic CVaR** at confidence level $p$ — two cases on the sign of $p - 0.5$:

If $p < 0.5$:

$$\text{CVaR}_p[\tilde{A}] = b_2 + \frac{[\,2p(1-(2p)^k) + k(2p-1)\,]\alpha + \beta}{2(k+1)(1-p)}$$

If $p \geq 0.5$:

$$\text{CVaR}_p[\tilde{A}] = \frac{\alpha + \beta - k\beta\,(2(1-p))^{1/k}}{k+1}$$

---

**Eq. 7 — Credibilistic Skewness** (let $\rho_2 = b_2 - e$):

$$S[\tilde{A}] = \rho_2^3 + \frac{3}{2}(\beta \cdot \Xi_1 - k\alpha \cdot \Xi_2)$$

where $\Xi_1 = \rho_2^2 c_1 + 2\beta\rho_2 c_2 + 2\beta^2 c_3$ and $\Xi_2 = \rho_2^2 c'_1 - 2k\alpha\rho_2 c'_2 + 2k^2\alpha^2 c'_3$.

---

**Eq. 8 — Credibilistic Semikurtosis** — two cases on the sign of $e - b_2$:

- If $e \leq b_2$: &nbsp; $\text{SK}[\tilde{A}] = 12k^4\, \rho_1^{1/k+4}\, c'_4 \,/\, \alpha^{1/k}$
- If $e > b_2$: &nbsp; $\text{SK}[\tilde{A}] = 2\beta\,\Omega_1 - 2k\alpha\,\Omega_2 + \rho_2^4 - 12\,\rho_3^{k+4}\, c_4 \,/\, \beta^k$

---

## MOGA — How the Optimiser Works

Each portfolio is encoded as a weight vector $\mathbf{x} \in \mathbb{R}^n$ where exactly $\kappa=5$ entries are non-zero (cardinality constraint).

### Portfolio constraints
| Constraint | Value |
|-----------|-------|
| Budget | $\sum x_i = 1$ |
| Bounds | $0.08 \leq x_i \leq 0.30$ for active assets |
| Cardinality | Exactly 5 active assets |
| No short-selling | $x_i \geq 0$ |

### Algorithm flow (Algorithm 1 of the paper)

```
For each of R=30 independent runs:
    1. Initialise population of 180 random feasible portfolios
    2. For 2000 generations:
        a. CCBEX crossover (Appendix B.1) — bounded exponential crossover on active assets
        b. Swap mutation (Appendix B.2) — swap one active asset with one inactive asset
        c. Power mutation (Appendix B.3) — perturb one active asset weight
        d. Repair mechanism (Appendix B.4) — project back to budget constraint
        e. NSGA-II selection — non-dominated sorting + crowding distance
    3. Extract Pareto front
Pool all 30 Pareto fronts → extract global Pareto front
Filter: keep only portfolios with mean ≥ 2% AND skewness ≥ 0
K-medoids clustering (k=25) → select 25 diverse representative portfolios
```

### Crossover: CCBEX (Cardinality Constrained Bounded Exponential Crossover)
Only operates on genes where **both parents are active**. Uses an exponential distribution to generate bounded offspring weights — respecting $[l, u]$ bounds while exploring the search space efficiently.

### Mutation operators
- **Swap mutation**: replaces one randomly chosen active asset with a randomly chosen inactive one — explores different asset combinations
- **Power mutation**: perturbs one active asset's weight using a power-law distribution — fine-tunes allocations

### Repair mechanism
After crossover/mutation, the budget constraint $\sum x_i = 1$ may be violated. The repair mechanism scales weights proportionally back to $[l, u]$ bounds while maintaining the sum-to-one constraint.

### NSGA-II selection
From the combined parent + offspring pool (360 solutions):
1. Non-dominated sorting → fronts $F_1, F_2, \ldots$
2. Fill next generation from $F_1$ first, then $F_2$, etc.
3. When a front partially fits: rank by crowding distance (prefer solutions in less crowded regions of objective space)

---

## Project Structure

```
.
├── credibilistic_portfolio.ipynb   # Stage 1: data pipeline & stock selection
├── credibilistic_moga.ipynb        # Stage 2: CTFN fitting + MOGA (3-model baseline)
├── EVaR_Improvement_Stage1.ipynb   # Stage 2 + EVaR (4-model variant — see "Improvements")
├── Improvement_Papers/             # Reference PDFs for further improvement ideas
├── data/
│   ├── nse_train.csv               # NSE training returns (72 × 18, decimal fractions)
│   ├── nse_test.csv                # NSE test returns    (24 × 18)
│   ├── nyse_train.csv              # NYSE training returns
│   ├── nyse_test.csv               # NYSE test returns
│   ├── nifty_stock_key.json        # S1–S18 → ticker/name mapping (NSE)
│   ├── djia_stock_key.json         # S1–S18 → ticker/name mapping (NYSE)
│   ├── nse_prices.csv              # Raw monthly prices (NSE)
│   ├── nyse_prices.csv             # Raw monthly prices (NYSE)
│   ├── nse_returns.csv             # Full return series (NSE)
│   ├── nyse_returns.csv            # Full return series (NYSE)
│   ├── nifty_completeness_report.csv
│   ├── djia_completeness_report.csv
│   └── *.png                       # Plots: heatmaps, cumulative returns
└── README.md
```


## Improvements Beyond the Paper

The baseline (`credibilistic_moga.ipynb`) replicates Mandal et al. (2024) exactly. On top of that, this repo explores improvements drawn from related literature — each in its own notebook so the baseline is preserved as a fallback. The naming convention is `<Improvement>_Stage<N>.ipynb`.

### Stage 1 — EVaR (Entropic Value at Risk)

**File:** `EVaR_Improvement_Stage1.ipynb`
**Reference:** Chennaf, S. & Ben Amor, J. (2023). *Entropic value at risk to find the optimal uncertain random portfolio.* Soft Computing, 27, 15185–15197. [doi:10.1007/s00500-023-08547-5](https://doi.org/10.1007/s00500-023-08547-5)

**Idea:** Replace CVaR (Model III) with EVaR — a coherent risk measure that bounds CVaR from above ($\text{VaR} \leq \text{CVaR} \leq \text{EVaR}$) and is more sensitive to extreme losses:

$$\text{EVaR}_\beta(X) = \inf_{t>0}\left[\, t^{-1}\ln\frac{M_X(t)}{1-\beta} \,\right]$$

where $M_X(t) = E[\exp(tX)]$ is the moment generating function.

**Implementation note:** the Chennaf 2023 paper does **not** give a closed-form EVaR for a CTFN, so EVaR is computed empirically from the portfolio return time series via the MGF definition (`scipy.optimize.minimize_scalar` on losses = −returns, with a log-sum-exp trick for numerical stability). Mean / skewness / semikurtosis remain closed-form fuzzy moments. EVaR appears as a new **Model IV** alongside the original three models so the swap can be compared head-to-head.

**Quick-mode results** (`pop=60, gen=200, runs=3, 5 representative portfolios per market`) — averaged across representative portfolios on the 24-month test window:

| Market | Metric | Model III (CVaR) | Model IV (EVaR) | Δ |
|--------|--------|------------------|-----------------|----|
| NSE  | Final cumulative return | 0.4320 | **0.4676** | +8.2% |
| NSE  | Sharpe ratio            | 0.4099 | **0.4485** | +9.4% |
| NSE  | Max drawdown            | −0.1082 | −0.1110   | slightly worse |
| NYSE | Final cumulative return | 0.2502 | **0.2779** | +11.1% |
| NYSE | Sharpe ratio            | 0.2336 | **0.2567** | +9.9% |
| NYSE | Max drawdown            | −0.1051 | −0.1135   | slightly worse |

EVaR-optimised portfolios delivered higher cumulative return and Sharpe on out-of-sample data on **both** markets. The marginal increase in max drawdown is consistent with EVaR being a more aggressive bound on loss severity rather than loss frequency. A full-replication run (`pop=180, gen=2000, runs=30, 25 reps`) is recommended before drawing final conclusions.

**Reproduce:** open `EVaR_Improvement_Stage1.ipynb`, ensure `QUICK_TEST = True`, and run all cells (~3–5 minutes). Cell §12 prints the head-to-head CVaR vs EVaR table; output plots are saved to `data/{nse,nyse}_test_cumulative_evar.png`.

---

## Setup

```bash
# Clone
git clone https://github.com/PP112004/Credibilistic_Portfolio_Optimisation_Using_Higher_Order_Fuzzy_Moments.git
cd Credibilistic_Portfolio_Optimisation_Using_Higher_Order_Fuzzy_Moments

# Environment
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

# Dependencies
pip install numpy pandas scipy scikit-learn matplotlib yfinance jupyter
```

**Important:** Do **not** install `scikit-learn-extra` — it is incompatible with NumPy 2.x. This project uses standard `sklearn.cluster.KMeans` as a drop-in replacement.

---

## Running

### Quick test (~2.5 minutes baseline / ~3.5 minutes EVaR variant)
Open either `credibilistic_moga.ipynb` (3-model baseline) or `EVaR_Improvement_Stage1.ipynb` (4-model variant) in Jupyter and ensure the relevant cell has:
```python
QUICK_TEST = True   # pop=60, gen=200, runs=3
```
Then run all cells.

### Full replication (~4–8 hours) — reproduces paper Tables 6–11
Change cell `c009` to:
```python
QUICK_TEST = False  # pop=180, gen=2000, runs=30
```
Run overnight from terminal (survives closing Jupyter):
```bash
source venv/bin/activate
nohup jupyter nbconvert --to notebook --execute credibilistic_moga.ipynb \
  --output credibilistic_moga_full.ipynb \
  --ExecutePreprocessor.timeout=36000 > moga_run.log 2>&1 &

tail -f moga_run.log   # monitor progress
```

---

## Results Structure

Each model × market combination produces a table of 25 representative portfolios, each with:

| Column | Description |
|--------|-------------|
| `b1, b2, b3` | CTFN parameters (left anchor, median, right anchor) |
| `k` | CTFN shape parameter |
| `cp, mp` | Crossover and mutation probabilities (from grid search) |
| `Mean` | Credibilistic expected monthly return |
| `SV / MASD / CVaR` | Risk measure (model-dependent) |
| `Skewness` | Credibilistic skewness (positive = right-skewed, desirable) |
| `SemiKurt` | Credibilistic semikurtosis (lower = lighter left tail) |

---

## Dependencies

| Package | Version tested | Purpose |
|---------|---------------|---------|
| `numpy` | 2.4.4 | Numerical computation |
| `pandas` | 2.x | Data handling |
| `scipy` | 1.x | EVaR inner minimisation (`minimize_scalar`) |
| `scikit-learn` | 1.x | KMeans clustering |
| `matplotlib` | 3.x | Plots |
| `yfinance` | 0.2.x | Price download (portfolio notebook only) |
| `jupyter` | — | Notebook execution |

---

## Citation

If you use this replication in your work, please cite the original paper:

```bibtex
@article{mandal2024credibilistic,
  title={Credibilistic portfolio optimization with higher-order moments using coherent triangular fuzzy numbers},
  author={Mandal, Prasenjit Kumar and Thakur, Manoj and Mittal, Garima},
  journal={Applied Soft Computing},
  volume={151},
  pages={111155},
  year={2024},
  publisher={Elsevier},
  doi={10.1016/j.asoc.2023.111155}
}
```
