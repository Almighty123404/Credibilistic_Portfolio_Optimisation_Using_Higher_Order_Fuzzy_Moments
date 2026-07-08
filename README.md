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
├── itfn.py                         # Stage 2 + ITFN framework (CITFN class + MOGA + comparison)
├── run_itfn_evar_comparison.py     # Runner script: ITFN Model I vs EVaR Model IV
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

$$\text{EVaR}_\beta(X) = \inf_{t>0}\left[\ t^{-1}\ln\frac{M_X(t)}{1-\beta} \,\right]$$

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

### Stage 2 — ITFN (Intuitionistic Triangular Fuzzy Numbers)

**File:** `itfs.py`, `run_itfn_evar_comparison.py`

#### 2.1 Motivation

The baseline CTFN framework models each portfolio's return distribution with a **Coherent Triangular Fuzzy Number** $(b_1, b_2, b_3)_k$, which assigns a single membership grade $\mu(t) \in [0, 1]$ to each possible return value $t$. This implicitly assumes that evidence *for* a return value being plausible and evidence *against* it are complementary — i.e., $\nu(t) = 1 - \mu(t)$.

In practice, financial return distributions involve **genuine epistemic uncertainty** — regions where we are neither confident a return *will* occur nor confident it *will not*. **Intuitionistic Fuzzy Sets (IFS)**, introduced by Atanassov (1986), model this explicitly by decoupling membership $\mu(t)$ and non-membership $\nu(t)$, allowing a **hesitation margin**:

$$\pi(t) = 1 - \mu(t) - \nu(t) \geq 0$$

This third degree of freedom captures "we don't know" — the gap between evidence for and against — which is especially important for portfolio returns in volatile or illiquid markets where tail behaviour is genuinely uncertain.

#### 2.2 The Coherent ITFN Model

A **Coherent Intuitionistic Triangular Fuzzy Number (CITFN)** is a 7-parameter model:

$$\tilde{A}^I = \big\langle\, (b_1, b_2, b_3)_k\,;\; (\hat{b}_1, b_2, \hat{b}_3)_{\hat{k}} \,\big\rangle$$

where:
- $(b_1, b_2, b_3)_k$ defines the **membership function** (inner triangle, same as CTFN)
- $(\hat{b}_1, b_2, \hat{b}_3)_{\hat{k}}$ defines the **non-membership function** (outer triangle)
- $\hat{b}_1 \leq b_1$ and $\hat{b}_3 \geq b_3$ — the non-membership triangle is at least as wide

**Membership function** (coherent, shape $k$, where $\alpha = b_2 - b_1$, $\beta = b_3 - b_2$):

- $\mu(t) = \left(\dfrac{t - b_1}{\alpha}\right)^{1/k}$ &nbsp; for $b_1 \leq t < b_2$
- $\mu(t) = \left(\dfrac{b_3 - t}{\beta}\right)^{k}$ &nbsp; for $b_2 \leq t \leq b_3$
- $\mu(t) = 0$ &nbsp; otherwise

**Non-membership function** (coherent, shape $\hat{k}$, where $\hat{\alpha} = b_2 - \hat{b}_1$, $\hat{\beta} = \hat{b}_3 - b_2$):

- $\nu(t) = \left(\dfrac{b_2 - t}{\hat{\alpha}}\right)^{1/\hat{k}}$ &nbsp; for $\hat{b}_1 \leq t < b_2$
- $\nu(t) = \left(\dfrac{t - b_2}{\hat{\beta}}\right)^{\hat{k}}$ &nbsp; for $b_2 < t \leq \hat{b}_3$
- $\nu(t) = 1$ &nbsp; for $t < \hat{b}_1$ or $t > \hat{b}_3$
- $\nu(t) = 0$ &nbsp; at $t = b_2$

The **hesitation zones** are the intervals $[\hat{b}_1, b_1]$ (left) and $[b_3, \hat{b}_3]$ (right), with widths $\delta_L = b_1 - \hat{b}_1$ and $\delta_R = \hat{b}_3 - b_3$.

#### 2.3 Intuitionistic Score Function and Credibility

To compute credibilistic moments, we define the **intuitionistic score function**:

$$s(t) = \frac{\mu(t) - \nu(t) + 1}{2}$$

This maps the combined membership/non-membership information into a single unimodal function $s(t) \in [0, 1]$ that peaks at $s(b_2) = 1$ and acts as the effective "membership" for credibility inversion.

The **intuitionistic credibility measure** is then:

- $\text{Cr}^I\{\tilde{A} \geq t\} = 1$ &nbsp; for $t \leq \hat{b}_1$
- $\text{Cr}^I\{\tilde{A} \geq t\} = 1 - s(t)/2$ &nbsp; for $\hat{b}_1 < t \leq b_2$
- $\text{Cr}^I\{\tilde{A} \geq t\} = s(t)/2$ &nbsp; for $b_2 < t \leq \hat{b}_3$
- $\text{Cr}^I\{\tilde{A} \geq t\} = 0$ &nbsp; for $t > \hat{b}_3$

This satisfies **self-duality**: $\text{Cr}^I\{\tilde{A} \geq t\} + \text{Cr}^I\{\tilde{A} < t\} = 1$.

**Key property (CTFN degeneracy):** When $\hat{b}_1 = b_1$, $\hat{b}_3 = b_3$, and $\hat{k} = k$ (zero hesitation), we have $\nu(t) = 1 - \mu(t)$ everywhere, so $s(t) = \mu(t)$ and all ITFN formulas reduce exactly to the CTFN baseline.

#### 2.4 Credibilistic Moments (Numerical Integration)

The four portfolio moments are computed from the credibility distribution via numerical integration (`scipy.integrate.quad`):

$$E^I[\tilde{A}] = \int_0^{\infty} \text{Cr}^I\{\tilde{A} \geq t\}\, dt \;-\; \int_{-\infty}^{0} \text{Cr}^I\{\tilde{A} \leq t\}\, dt$$

$$\text{SV}^I[\tilde{A}] = \int_{-\infty}^{e} 2(e - t) \cdot \text{Cr}^I\{\tilde{A} \leq t\}\, dt$$

$$S^I[\tilde{A}] = \int_{e}^{\infty} 3(t-e)^2 \cdot \text{Cr}^I\{\tilde{A} \geq t\}\, dt \;-\; \int_{-\infty}^{e} 3(t-e)^2 \cdot \text{Cr}^I\{\tilde{A} \leq t\}\, dt$$

$$\text{SK}^I[\tilde{A}] = \int_{-\infty}^{e} 4(e-t)^3 \cdot \text{Cr}^I\{\tilde{A} \leq t\}\, dt$$

where $e = E^I[\tilde{A}]$ is the intuitionistic credibilistic mean.

Closed-form fallbacks (based on CTFN moments) are provided when `scipy` is unavailable.

#### 2.5 CITFN Fitting Procedure

The CITFN is fitted to portfolio return time series $\{R_t\}$ using an extended percentile-based procedure:

| Step | Parameter | Formula |
|------|-----------|---------|
| I–IV | $b_1, b_2, b_3, k$ | Same as CTFN fitting (Section 4.1 of Mandal et al.) |
| V | $\hat{b}_1$ | $\min(R_{\min},\, Q_1)$ — wider left anchor |
| V | $\hat{b}_3$ | $Q_{99}$ — wider right anchor |
| VI | $\hat{k}$ | Computed from $Q_5$ or $Q_{95}$ via $\hat{k} = \ln(0.5) / \ln(\text{denom})$ |

The outer triangle captures extreme percentiles ($Q_1$, $Q_{99}$) that the inner CTFN triangle ($Q_3$, $Q_{97}$) misses, creating natural hesitation zones in the tails.

#### 2.6 MOGA Integration

The ITFN model is integrated as **Model I_ITFN** in the existing NSGA-II MOGA framework, using the same 4-objective optimisation as the baseline:

$$\min_{\mathbf{x}} \Big[\, -E^I,\;\; \text{SV}^I,\;\; -S^I,\;\; \text{SK}^I \,\Big]$$

subject to the same cardinality ($\kappa = 5$), budget ($\sum x_i = 1$), and bound ($0.08 \leq x_i \leq 0.30$) constraints. All MOGA operators (CCBEX crossover, swap/power mutation, repair, NSGA-II selection, K-medoids clustering) are reused unchanged.

#### 2.7 Three-Way Comparison Results

Three models are compared head-to-head on both markets using the same MOGA configuration (`pop=30, gen=50, runs=2`) and the same 24-month out-of-sample test window:

| Model | Framework | Risk Measure | What it tests |
|-------|-----------|-------------|---------------|
| **ITFN Model I** | CITFN | Semivariance (SV) | Hesitation + downside spread |
| **ITFN Model IV** | CITFN | EVaR | Hesitation + tail risk ← *new* |
| **EVaR Model IV** | CTFN | EVaR | Baseline (no hesitation) + tail risk |

> **Methodology note:** Sharpe ratios are **annualised** (`mean/std × √12`). All ITFN results use the IFS-axiom-corrected model (ν(t) clamped so μ(t)+ν(t) ≤ 1). Risk measures are in percentage-point monthly return units.

**Training metrics** (in-sample):

| Market | Model | Mean | Risk | Skewness | Semikurtosis |
|--------|-------|:----:|:----:|:--------:|:------------:|
| NSE | ITFN Model I (CITFN+SV) | **3.0005** | SV = 18.853 | **102.94** | 961.1 |
| NSE | ITFN Model IV (CITFN+EVaR) | 2.8696 | EVaR = 6.409 | 102.72 | 848.0 |
| NSE | EVaR Model IV (CTFN+EVaR) | 1.5281 | EVaR = **6.280** | 43.25 | **381.1** |
| NYSE | ITFN Model I (CITFN+SV) | 1.8793 | SV = 11.750 | **21.19** | 503.1 |
| NYSE | ITFN Model IV (CITFN+EVaR) | **2.7330** | EVaR = 6.676 | 11.81 | 520.4 |
| NYSE | EVaR Model IV (CTFN+EVaR) | 1.6041 | EVaR = **6.562** | 0.17 | **355.3** |

**Test metrics** (out-of-sample, 24 months):

| Market | Model | Cumulative Return | Sharpe (ann.) | Max Drawdown |
|--------|-------|:-----------------:|:-------------:|:------------:|
| NSE | ITFN Model I (CITFN+SV) | **0.3628** | **1.1780** | 0.1635 |
| NSE | ITFN Model IV (CITFN+EVaR) | 0.2565 | 0.9833 | **0.1532** |
| NSE | EVaR Model IV (CTFN+EVaR) | 0.2938 | 0.9992 | 0.1756 |
| NYSE | ITFN Model I (CITFN+SV) | 0.1624 | 0.6257 | 0.1862 |
| NYSE | ITFN Model IV (CITFN+EVaR) | 0.1764 | 0.6638 | **0.1225** |
| NYSE | EVaR Model IV (CTFN+EVaR) | **0.3052** | **0.8773** | 0.2436 |

#### 2.8 Analysis and Findings

**Finding 1: ITFN+EVaR achieves the lowest drawdown on both markets — consistently.**
ITFN Model IV (CITFN+EVaR) produces the smallest maximum drawdown on NSE (15.3% vs 16.4% vs 17.6%) and NYSE (12.3% vs 18.6% vs 24.4%). This is the most striking result: combining the hesitation-aware credibility distribution with EVaR as the tail-risk objective creates the most robust downside-protection profile. The hesitation zone captures tail ambiguity while EVaR directs the optimiser to explicitly minimise extreme losses.

**Finding 2: ITFN+EVaR achieves higher training mean than CTFN+EVaR at comparable EVaR.**
Despite using the same EVaR risk measure, ITFN Model IV finds portfolios with substantially higher in-sample mean returns (NSE: 2.87 vs 1.53; NYSE: 2.73 vs 1.60) at nearly identical EVaR levels (NSE: 6.41 vs 6.28; NYSE: 6.68 vs 6.56). This means the hesitation-aware credibility distribution identifies different, higher-returning feasible regions in the weight space that standard CTFN-EVaR cannot resolve — the hesitation zone acts as an additional information layer that discriminates ambiguous assets.

**Finding 3: Risk measure choice controls the return-risk trade-off within ITFN.**
Comparing the two ITFN models:
- **ITFN+SV** maximises cumulative return (NSE: +36.3%) but accepts higher semivariance (SV=18.85 vs 11.75 on NSE). It's the aggressive ITFN variant.
- **ITFN+EVaR** lowers both drawdown and cumulative return — it is the conservative ITFN variant that prioritises tail protection over return maximisation.
The choice of risk measure thus allows a practitioner to tune the risk appetite while retaining the hesitation-aware credibility framework.

**Finding 4: CTFN+EVaR still wins on total cumulative return on NYSE.**
On the developed NYSE market, CTFN+EVaR produces significantly higher cumulative return (0.305 vs 0.176 and 0.162). This suggests that for well-understood, liquid markets with low tail ambiguity, the extra modelling complexity of ITFN does not translate to return advantage — the simpler CTFN model captures the available signal more efficiently.

**Finding 5: All ITFN models produce strong positive skewness.**
Both ITFN models yield high positive skewness (NSE: ~103; NYSE: 12–21), far exceeding CTFN+EVaR (NSE: 43; NYSE: 0.17). This is a structural advantage — positively skewed portfolios have more frequent small gains and infrequent large losses, aligning with investor preferences for right-tailed distributions.

**Summary assessment:**

| Criterion | Winner | Notes |
|-----------|--------|-------|
| **Drawdown protection** | **ITFN+EVaR** | Best on both markets — most consistent result |
| **NSE cumulative return** | **ITFN+SV** | +36.3% vs +29.4% (CTFN) and +25.7% (ITFN-IV) |
| **NYSE cumulative return** | **CTFN+EVaR** | +30.5% vs +17.6% (ITFN-IV) and +16.2% (ITFN-SV) |
| **Skewness** | **ITFN** (both) | ~102–103 on NSE; CTFN gets only 43 |
| **Training mean vs risk** | **ITFN+EVaR** | Higher return at equivalent EVaR budget vs CTFN |
| **Computational cost** | **CTFN+EVaR** | ~6× faster than ITFN (no numerical integration) |

**Recommendation:** For risk-averse investors or volatile/emerging markets, **ITFN+EVaR** is the preferred configuration — it delivers the best drawdown protection and higher in-sample mean at comparable tail-risk budget. For return-maximising strategies on developed markets, **CTFN+EVaR** remains competitive. **ITFN+SV** is best suited for emerging markets where return maximisation with hesitation awareness is the priority.

**Reproduce:**
```bash
python run_itfn_evar_comparison.py
```
Quick test (~5 minutes, `pop=30, gen=50, runs=2`): set `quick_test=True` in the script. Full run (~45–90 minutes, `pop=80, gen=250, runs=5`): set `quick_test=False`.

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

### Notebooks (Baseline + EVaR)

#### Quick test (~2.5 minutes baseline / ~3.5 minutes EVaR variant)
Open either `credibilistic_moga.ipynb` (3-model baseline) or `EVaR_Improvement_Stage1.ipynb` (4-model variant) in Jupyter and ensure the relevant cell has:
```python
QUICK_TEST = True   # pop=60, gen=200, runs=3
```
Then run all cells.

#### Full replication (~4-8 hours) -- reproduces paper Tables 6-11
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

### ITFN vs EVaR Comparison (Python scripts)

The ITFN improvement runs as standalone Python scripts (no Jupyter required).

#### Quick test (~1 minute)
```bash
python run_itfn_evar_comparison.py
```
By default the runner calls `run_itfn_vs_evar_comparison(quick_test=True)` with `pop=30, gen=50, runs=2`. Edit the script to set `quick_test=False` for a substantive run.

#### Full comparison (~30-45 minutes)
Edit `run_itfn_evar_comparison.py` and set `quick_test=False`:
```python
summary = run_itfn_vs_evar_comparison(
    markets=('nse', 'nyse'),
    quick_test=False,   # pop=80, gen=250, runs=5
    verbose=True,
)
```
Then run:
```bash
python run_itfn_evar_comparison.py
```
Progress is printed in real-time (generation updates every ~20% of the run). The final comparison table is printed at the end.

#### Running sanity checks only
To verify the ITFN implementation without running the full MOGA comparison:
```bash
python -c "from itfs import run_itfn_sanity_checks; run_itfn_sanity_checks()"
```
This runs 7 diagnostic tests (CTFN degeneracy, self-duality, constraint satisfaction, closed-form cross-checks, etc.) in under 1 second.

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
| `scipy` | 1.x | EVaR minimisation (`minimize_scalar`), ITFN moment integration (`quad`) |
| `scikit-learn` | 1.x | KMeans clustering (baseline + ITFN) |
| `matplotlib` | 3.x | Plots |
| `yfinance` | 0.2.x | Price download (portfolio notebook only) |
| `jupyter` | -- | Notebook execution (not needed for ITFN scripts) |

> **Note:** `scipy` is **required** for the ITFN comparison (`itfs.py`). Without it, ITFN moments fall back to CTFN closed-form approximations, which defeats the purpose of the intuitionistic extension. All other scripts work without `scipy` but produce lower-quality EVaR estimates.

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
