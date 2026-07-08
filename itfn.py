"""
itfs.py — Coherent Intuitionistic Triangular Fuzzy Number (CITFN) Framework
=============================================================================

Extends the CTFN-based credibilistic portfolio optimisation framework
(Mandal, Thakur & Mittal, 2024) with Intuitionistic Triangular Fuzzy Numbers.

Key additions over CTFNs:
  - 7-parameter model: (b1, b2, b3)_k  for membership;  (b1_hat, b2, b3_hat)_k_hat for non-membership
  - Score-function credibility:  s(t) = (mu - nu + 1) / 2
  - 4-region credibility distribution (vs 3 for CTFNs)
  - IFS axiom enforced: nu(t) is clamped so that mu(t) + nu(t) <= 1 at every t
  - Additive moment decomposition:  Moment^I = Moment^CTFN + Delta_nu + Delta_pi

When b1_hat == b1, b3_hat == b3 and k_hat == k (zero hesitation, k=1 for
exact CTFN degeneracy), all ITFN formulas reduce to the CTFN special case.
"""

import sys
import time
import numpy as np
import pandas as pd
import warnings
from copy import deepcopy
from sklearn.cluster import KMeans

try:
    from scipy.integrate import quad as _quad
    from scipy.optimize import minimize_scalar as _minimize_scalar
    _HAS_SCIPY = True
    _HAS_SCIPY_OPT = True
except ImportError:
    _HAS_SCIPY = False
    _HAS_SCIPY_OPT = False


# ═══════════════════════════════════════════════════════════════════════════════
#  §1  HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def _c(k, j):
    """c_j = 1 / [(k+1)(k+2)...(k+j)]"""
    r = 1.0
    for i in range(1, j + 1):
        r *= 1.0 / (k + i)
    return r


def _cp(k, j):
    """c'_j = 1 / [(1+k)(1+2k)...(1+jk)]"""
    r = 1.0
    for i in range(1, j + 1):
        r *= 1.0 / (1.0 + i * k)
    return r


# ═══════════════════════════════════════════════════════════════════════════════
#  §2  CTFN BASELINE FUNCTIONS  (extracted from credibilistic_moga.ipynb)
# ═══════════════════════════════════════════════════════════════════════════════

def ctfn_mean(b1, b2, b3, k):
    """Eq. 3 — Credibilistic Mean:  E[Ã] = b2 + (β − kα) / (2(k+1))"""
    alpha = b2 - b1
    beta = b3 - b2
    return b2 + (beta - k * alpha) / (2 * (k + 1))


def ctfn_sv(b1, b2, b3, k, e=None):
    """Eq. 4 — Credibilistic Semivariance."""
    alpha = b2 - b1
    beta = b3 - b2
    if e is None:
        e = ctfn_mean(b1, b2, b3, k)
    c1 = _c(k, 1)
    c2 = _c(k, 2)
    cp2 = _cp(k, 2)
    rho2 = b2 - e

    if e <= b2:  # Case 1: b1 <= e <= b2
        rho1 = e - b1
        return k ** 2 * (rho1 ** (1 / k + 2)) * cp2 / (alpha ** (1 / k))
    else:  # Case 2: b2 < e <= b3
        rho3 = b3 - e
        Xi1 = -k * alpha * rho2 * c1 + k ** 2 * alpha ** 2 * cp2 + rho2 ** 2 + rho2 * beta * c1 + beta ** 2 * c2
        return Xi1 - (rho3 ** (k + 2)) * c2 / (beta ** k)


def ctfn_masd(b1, b2, b3, k):
    """Eq. 5 — Credibilistic MASD."""
    alpha = b2 - b1
    beta = b3 - b2
    if k * alpha >= beta:
        inner = 1 + (beta - k * alpha) / (2 * alpha * (k + 1))
        return k * alpha / (2 * (k + 1)) * (inner ** ((k + 1) / k))
    else:
        inner = 1 + (k * alpha - beta) / (2 * beta * (k + 1))
        return beta / (2 * (k + 1)) * (inner ** (k + 1))


def ctfn_cvar(b1, b2, b3, k, p=0.95):
    """Eq. 6 — Credibilistic CVaR."""
    alpha = b2 - b1
    beta = b3 - b2
    if p < 0.5:
        num = (2 * p * (1 - (2 * p) ** k) + k * (2 * p - 1)) * alpha + beta
        return b2 + num / (2 * (k + 1) * (1 - p))
    else:
        return (alpha + beta - k * beta * (2 * (1 - p)) ** (1 / k)) / (k + 1)


def ctfn_skewness(b1, b2, b3, k, e=None):
    """Eq. 7 — Credibilistic Skewness:  S[Ã] = ρ₂³ + (3/2)(β·Ξ₁ − kα·Ξ₂)"""
    alpha = b2 - b1
    beta = b3 - b2
    if e is None:
        e = ctfn_mean(b1, b2, b3, k)
    c1 = _c(k, 1)
    c2 = _c(k, 2)
    c3 = _c(k, 3)
    cp1 = _cp(k, 1)
    cp2 = _cp(k, 2)
    cp3 = _cp(k, 3)
    rho2 = b2 - e
    Xi1 = rho2 ** 2 * c1 + 2 * beta * rho2 * c2 + 2 * beta ** 2 * c3
    Xi2 = rho2 ** 2 * cp1 - 2 * k * alpha * rho2 * cp2 + 2 * k ** 2 * alpha ** 2 * cp3
    return rho2 ** 3 + 1.5 * (beta * Xi1 - k * alpha * Xi2)


def ctfn_sk(b1, b2, b3, k, e=None):
    """Eq. 8 — Credibilistic Semikurtosis."""
    alpha = b2 - b1
    beta = b3 - b2
    if e is None:
        e = ctfn_mean(b1, b2, b3, k)
    c1 = _c(k, 1)
    c2 = _c(k, 2)
    c3 = _c(k, 3)
    c4 = _c(k, 4)
    cp1 = _cp(k, 1)
    cp2 = _cp(k, 2)
    cp3 = _cp(k, 3)
    cp4 = _cp(k, 4)
    rho2 = b2 - e

    if e <= b2:  # Case 1: b1 <= e <= b2
        rho1 = e - b1
        return 12 * k ** 4 * (rho1 ** (1 / k + 4)) * cp4 / (alpha ** (1 / k))
    else:  # Case 2: b2 < e <= b3
        rho3 = b3 - e
        Omega1 = rho2 ** 3 * c1 + 3 * beta * rho2 ** 2 * c2 + 6 * beta ** 2 * rho2 * c3 + 6 * beta ** 3 * c4
        Omega2 = rho2 ** 3 * cp1 - 3 * k * alpha * rho2 ** 2 * cp2 + 6 * k ** 2 * alpha ** 2 * rho2 * cp3 - 6 * k ** 3 * alpha ** 3 * cp4
        return 2 * beta * Omega1 - 2 * k * alpha * Omega2 + rho2 ** 4 - 12 * (rho3 ** (k + 4)) * c4 / (beta ** k)


# ═══════════════════════════════════════════════════════════════════════════════
#  §3  COHERENT ITFN CLASS
# ═══════════════════════════════════════════════════════════════════════════════

class CoherentITFN:
    """Coherent Intuitionistic Triangular Fuzzy Number.

    Parameters
    ----------
    b1, b2, b3 : float
        Membership triangle (inner).   b1 <= b2 <= b3.
    k : float
        Shape parameter for membership function.
    b1_hat, b3_hat : float
        Non-membership triangle endpoints (outer).   b1_hat <= b1,  b3_hat >= b3.
    k_hat : float
        Shape parameter for non-membership function.
    """

    def __init__(self, b1, b2, b3, k, b1_hat, b3_hat, k_hat):
        # ── Store parameters ──────────────────────────────────────────────────
        self.b1 = b1
        self.b2 = b2
        self.b3 = b3
        self.k = k
        self.b1_hat = b1_hat
        self.b3_hat = b3_hat
        self.k_hat = k_hat

        # ── Derived quantities ────────────────────────────────────────────────
        self.alpha = b2 - b1           # membership left spread
        self.beta = b3 - b2            # membership right spread
        self.alpha_hat = b2 - b1_hat   # non-membership left spread (wider)
        self.beta_hat = b3_hat - b2    # non-membership right spread (wider)
        self.delta_L = b1 - b1_hat     # left hesitation zone width
        self.delta_R = b3_hat - b3     # right hesitation zone width

        # ── Validate ──────────────────────────────────────────────────────────
        EPS = 1e-12
        if b1_hat > b1 + EPS:
            raise ValueError(f"b1_hat ({b1_hat}) must be <= b1 ({b1})")
        if b3_hat < b3 - EPS:
            raise ValueError(f"b3_hat ({b3_hat}) must be >= b3 ({b3})")
        if b1 > b2 + EPS:
            raise ValueError(f"b1 ({b1}) must be <= b2 ({b2})")
        if b2 > b3 + EPS:
            raise ValueError(f"b2 ({b2}) must be <= b3 ({b3})")
        if k <= 0:
            raise ValueError(f"k ({k}) must be > 0")
        if k_hat <= 0:
            raise ValueError(f"k_hat ({k_hat}) must be > 0")

    # ──────────────────────────────────────────────────────────────────────────
    #  Membership / Non-membership / Score / Hesitation
    # ──────────────────────────────────────────────────────────────────────────

    def membership(self, t):
        """μ(t) — coherent membership with shape k."""
        if t < self.b1 or t > self.b3:
            return 0.0
        if self.b1 <= t < self.b2:
            if self.alpha < 1e-15:
                return 1.0
            return ((t - self.b1) / self.alpha) ** (1.0 / self.k)
        else:  # b2 <= t <= b3
            if self.beta < 1e-15:
                return 1.0
            return ((self.b3 - t) / self.beta) ** self.k

    def nonmembership(self, t):
        """ν(t) — coherent non-membership with shape k_hat.

        The raw non-membership value is clamped to (1 − μ(t)) so that the
        IFS axiom  μ(t) + ν(t) ≤ 1  is satisfied at every point.  This is
        necessary for k ≠ 1 where the coherent exponents can otherwise
        produce μ + ν > 1.
        """
        if t < self.b1_hat or t > self.b3_hat:
            return 1.0
        if t == self.b2:
            return 0.0
        if self.b1_hat <= t < self.b2:
            if self.alpha_hat < 1e-15:
                raw_nu = 0.0
            else:
                raw_nu = ((self.b2 - t) / self.alpha_hat) ** (1.0 / self.k_hat)
        else:  # b2 < t <= b3_hat
            if self.beta_hat < 1e-15:
                raw_nu = 0.0
            else:
                raw_nu = ((t - self.b2) / self.beta_hat) ** self.k_hat
        # Enforce IFS axiom: mu(t) + nu(t) <= 1
        mu = self.membership(t)
        return min(raw_nu, 1.0 - mu)

    def score_membership(self, t):
        """s(t) = (μ(t) − ν(t) + 1) / 2  (intuitionistic score function)."""
        return (self.membership(t) - self.nonmembership(t) + 1.0) / 2.0

    def hesitation(self, t):
        """π(t) = 1 − μ(t) − ν(t) ≥ 0  (hesitation margin, IFS-axiom guaranteed)."""
        return max(0.0, 1.0 - self.membership(t) - self.nonmembership(t))

    # ──────────────────────────────────────────────────────────────────────────
    #  Five-region Intuitionistic Credibility Distribution
    # ──────────────────────────────────────────────────────────────────────────

    def credibility_geq(self, t):
        """Cr^I{A >= t} — Liu-Liu credibility inversion applied to score function s(t).

        s(t) = (mu(t) - nu(t) + 1) / 2  is unimodal, peaking at b2 where s=1.

        Credibility inversion for unimodal membership s:
          Left  (s increasing, t <= b2):  Cr{A >= t} = 1 - s(t)/2
          Right (s decreasing, t >  b2):  Cr{A >= t} = s(t)/2

        Both sides give Cr = 0.5 at t = b2 (continuous).
        Reduces to CTFN credibility when nu = 1 - mu.
        """
        s = self.score_membership(t)

        if t <= self.b1_hat:
            return 1.0  # s(t) = 0 for t < b1_hat, so 1 - 0/2 = 1
        elif t <= self.b2:
            # s is increasing on [b1_hat, b2]
            return 1.0 - s / 2.0
        elif t <= self.b3_hat:
            # s is decreasing on (b2, b3_hat]
            return s / 2.0
        else:
            return 0.0  # s(t) = 0 for t > b3_hat, so 0/2 = 0

    def credibility_leq(self, t):
        """Cr^I{Ã <= t} = 1 − Cr^I{Ã >= t}  (self-duality)."""
        return 1.0 - self.credibility_geq(t)

    # ──────────────────────────────────────────────────────────────────────────
    #  Intuitionistic Credibilistic Mean  (Numerical integration — reliable)
    # ──────────────────────────────────────────────────────────────────────────

    def mean(self):
        """E^I[Ã]  — computed by numerical integration of the credibility distribution.

        E^I = ∫_0^∞ Cr^I{Ã >= t} dt  −  ∫_{-∞}^0 Cr^I{Ã <= t} dt
        """
        if not _HAS_SCIPY:
            return self._mean_closed()

        # Positive part: ∫_0^∞ Cr^I{Ã >= t} dt
        # The credibility is 0 for t > b3_hat, so integrate [0, b3_hat]
        # But if b1_hat < 0, there's also a contribution from [b1_hat, 0]
        # handled by the negative part formula.
        # Actually, use the standard formula directly:
        # E = ∫_0^∞ Cr{>=t} dt - ∫_{-∞}^0 (1 - Cr{>=t}) dt
        #   = ∫_0^∞ Cr{>=t} dt - ∫_{-∞}^0 1 dt + ∫_{-∞}^0 Cr{>=t} dt
        #   = ∫_{-∞}^∞ Cr{>=t} dt - ∫_{-∞}^0 1 dt   ... but that diverges
        # Better: E = ∫_0^∞ Cr{>=t} dt - ∫_{-∞}^0 Cr{<=t} dt

        pos_part, _ = _quad(self.credibility_geq, 0, max(self.b3_hat, 0) + 1e-10,
                            limit=200, points=[self.b1_hat, self.b1, self.b2, self.b3, self.b3_hat])
        neg_part, _ = _quad(self.credibility_leq, min(self.b1_hat, 0) - 1e-10, 0,
                            limit=200, points=[self.b1_hat, self.b1, self.b2, self.b3, self.b3_hat])
        return pos_part - neg_part

    def _mean_closed(self):
        """Closed-form approximation of E^I[Ã].

        For the case b1 <= e <= b2 and symmetric-ish hesitation:
        E^I = b2 + (β − kα)/(2(k+1))  +  hesitation correction
        """
        k = self.k
        kh = self.k_hat
        alpha, beta = self.alpha, self.beta
        ah, bh = self.alpha_hat, self.beta_hat
        dL, dR = self.delta_L, self.delta_R

        # CTFN base mean
        e_ctfn = self.b2 + (beta - k * alpha) / (2.0 * (k + 1.0))

        # Hesitation zone corrections (from integration over [b1_hat, b1] and [b3, b3_hat])
        # Contribution from [b1_hat, b1]:
        # ∫ (1/2)[1 + ((b2-t)/ah)^(1/kh)] dt  over [b1_hat, b1]
        # = dL/2 + (kh/(2(kh+1))) * ah * [((b2-b1_hat)/ah)^(1/kh+1) - ((b2-b1)/ah)^(1/kh+1)]
        if dL > 1e-15 and ah > 1e-15:
            exp_l = 1.0 / kh + 1.0
            term_l = (kh / (2.0 * (kh + 1.0))) * ah * (
                (ah / ah) ** exp_l - (alpha / ah) ** exp_l
            )
            left_correction = dL / 2.0 + term_l
        else:
            left_correction = 0.0

        # Contribution from [b3, b3_hat]:
        # ∫ (1/2)[1 - ((t-b2)/bh)^kh] dt  over [b3, b3_hat]
        # = dR/2 - (1/(2*(kh+1))) * bh * [((b3_hat-b2)/bh)^(kh+1) - ((b3-b2)/bh)^(kh+1)]
        if dR > 1e-15 and bh > 1e-15:
            exp_r = kh + 1.0
            term_r = (1.0 / (2.0 * (kh + 1.0))) * bh * (
                (bh / bh) ** exp_r - (beta / bh) ** exp_r
            )
            right_correction = dR / 2.0 - term_r
        else:
            right_correction = 0.0

        # Non-membership correction on inner triangle [b1, b2]:
        # Additional ∫ (1/2)((b2-t)/ah)^(1/kh) dt  over [b1, b2]
        if ah > 1e-15:
            exp_inner = 1.0 / kh + 1.0
            inner_left_nu = (kh / (2.0 * (kh + 1.0))) * ah * (
                (alpha / ah) ** exp_inner  # at t=b1: (b2-b1)/ah = alpha/ah
            )
            # This is the difference: integral from b1 to b2 of (1/2)((b2-t)/ah)^(1/kh) dt
            # = (kh/(2(kh+1))) * ah * [(alpha/ah)^(1/kh+1) - 0]
        else:
            inner_left_nu = 0.0

        # Non-membership correction on inner triangle [b2, b3]:
        # Additional ∫ -(1/2)((t-b2)/bh)^kh dt  over [b2, b3]
        if bh > 1e-15:
            exp_inner_r = kh + 1.0
            inner_right_nu = -(1.0 / (2.0 * (kh + 1.0))) * bh * (
                (beta / bh) ** exp_inner_r
            )
        else:
            inner_right_nu = 0.0

        # The full ITFN mean = CTFN mean + all nu/hesitation corrections
        # The corrections affect the integral of Cr^I vs Cr^CTFN
        return e_ctfn + left_correction + right_correction + inner_left_nu + inner_right_nu

    # ──────────────────────────────────────────────────────────────────────────
    #  Intuitionistic Credibilistic Semivariance  (Numerical)
    # ──────────────────────────────────────────────────────────────────────────

    def semivariance(self, e=None):
        """SV^I[Ã] = ∫_{-∞}^{e} 2(e − t) · Cr^I{Ã ≤ t} dt"""
        if e is None:
            e = self.mean()

        if not _HAS_SCIPY:
            return self._semivariance_closed(e)

        lower = min(self.b1_hat, e) - 1e-10

        def integrand(t):
            return 2.0 * (e - t) * self.credibility_leq(t)

        pts = [p for p in [self.b1_hat, self.b1, self.b2, self.b3, self.b3_hat] if lower < p < e]
        result, _ = _quad(integrand, lower, e, limit=200, points=pts)
        return max(result, 0.0)

    def _semivariance_closed(self, e):
        """Fallback closed-form semivariance when scipy is unavailable.
        Uses CTFN SV as base + corrections.
        """
        # For robustness, just return CTFN SV (exact when hesitation=0)
        return ctfn_sv(self.b1, self.b2, self.b3, self.k, e)

    # ──────────────────────────────────────────────────────────────────────────
    #  Intuitionistic Credibilistic Skewness  (Numerical)
    # ──────────────────────────────────────────────────────────────────────────

    def skewness(self, e=None):
        """S^I[Ã] = ∫_e^∞ 3(t−e)² Cr^I{Ã≥t} dt  −  ∫_{-∞}^e 3(t−e)² Cr^I{Ã≤t} dt"""
        if e is None:
            e = self.mean()

        if not _HAS_SCIPY:
            return self._skewness_closed(e)

        def integrand_pos(t):
            return 3.0 * (t - e) ** 2 * self.credibility_geq(t)

        def integrand_neg(t):
            return 3.0 * (e - t) ** 2 * self.credibility_leq(t)

        upper = max(self.b3_hat, e) + 1e-10
        lower = min(self.b1_hat, e) - 1e-10

        pts_pos = [p for p in [self.b1_hat, self.b1, self.b2, self.b3, self.b3_hat] if e < p < upper]
        pts_neg = [p for p in [self.b1_hat, self.b1, self.b2, self.b3, self.b3_hat] if lower < p < e]

        pos_part, _ = _quad(integrand_pos, e, upper, limit=200, points=pts_pos)
        neg_part, _ = _quad(integrand_neg, lower, e, limit=200, points=pts_neg)
        return pos_part - neg_part

    def _skewness_closed(self, e):
        """Fallback when scipy unavailable."""
        return ctfn_skewness(self.b1, self.b2, self.b3, self.k, e)

    # ──────────────────────────────────────────────────────────────────────────
    #  Intuitionistic Credibilistic Semikurtosis  (Numerical)
    # ──────────────────────────────────────────────────────────────────────────

    def semikurtosis(self, e=None):
        """SK^I[Ã] = ∫_{-∞}^e 4(e−t)³ Cr^I{Ã≤t} dt"""
        if e is None:
            e = self.mean()

        if not _HAS_SCIPY:
            return self._semikurtosis_closed(e)

        lower = min(self.b1_hat, e) - 1e-10

        def integrand(t):
            return 4.0 * (e - t) ** 3 * self.credibility_leq(t)

        pts = [p for p in [self.b1_hat, self.b1, self.b2, self.b3, self.b3_hat] if lower < p < e]
        result, _ = _quad(integrand, lower, e, limit=200, points=pts)
        return max(result, 0.0)

    def _semikurtosis_closed(self, e):
        """Fallback when scipy unavailable."""
        return ctfn_sk(self.b1, self.b2, self.b3, self.k, e)

    # ──────────────────────────────────────────────────────────────────────────
    #  Hesitation Diagnostics
    # ──────────────────────────────────────────────────────────────────────────

    def total_hesitation(self, n_points=1000):
        """∫π(t) dt over [b1_hat, b3_hat]  (area of the hesitation zone).
        Uses numerical integration if scipy available, else trapezoidal.
        """
        if _HAS_SCIPY:
            pts = [self.b1_hat, self.b1, self.b2, self.b3, self.b3_hat]
            result, _ = _quad(self.hesitation, self.b1_hat, self.b3_hat,
                              limit=200, points=[p for p in pts if self.b1_hat < p < self.b3_hat])
            return result

        ts = np.linspace(self.b1_hat, self.b3_hat, n_points)
        hs = np.array([self.hesitation(t) for t in ts])
        return np.trapz(hs, ts)

    def hesitation_asymmetry(self):
        """(δ_L − δ_R) / (δ_L + δ_R).  In [-1, 1].
        Returns 0 if both hesitation zones are zero (CTFN case).
        """
        total = self.delta_L + self.delta_R
        if total < 1e-15:
            return 0.0
        return (self.delta_L - self.delta_R) / total

    def plot_membership(self, ax=None, n_points=500, title=None):
        """Plot μ(t), ν(t), π(t) on the same axes."""
        import matplotlib.pyplot as plt
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(10, 5))

        ts = np.linspace(self.b1_hat - 0.01, self.b3_hat + 0.01, n_points)
        mus = [self.membership(t) for t in ts]
        nus = [self.nonmembership(t) for t in ts]
        pis = [self.hesitation(t) for t in ts]

        ax.plot(ts, mus, 'b-', linewidth=2, label='μ(t) — membership')
        ax.plot(ts, nus, 'r-', linewidth=2, label='ν(t) — non-membership')
        ax.fill_between(ts, 0, pis, alpha=0.2, color='green', label='π(t) — hesitation')
        ax.axhline(y=0, color='gray', linewidth=0.5)
        ax.axhline(y=1, color='gray', linewidth=0.5)
        ax.set_xlabel('t')
        ax.set_ylabel('Degree')
        ax.set_ylim(-0.05, 1.15)
        ax.legend(loc='upper right')
        if title:
            ax.set_title(title)
        else:
            ax.set_title(f'CITFN: ({self.b1:.4f}, {self.b2:.4f}, {self.b3:.4f})_{{k={self.k:.3f}}}  '
                         f'({self.b1_hat:.4f}, {self.b2:.4f}, {self.b3_hat:.4f})_{{k_hat={self.k_hat:.3f}}}')
        return ax


# ==============================================================================
#  Section 4  FITTING FUNCTIONS
# ==============================================================================

def portfolio_returns(weights, z, R_matrix):
    """Compute time-series of portfolio returns.
    weights : (n,) array of asset weights
    z       : (n,) binary array (1=active, 0=inactive)
    R_matrix: (T, n) array — individual asset returns
    Returns : (T,) array
    """
    return R_matrix @ (weights * z)


def fit_ctfn(port_rets, u=None, rng=None):
    """Steps I–IV of Section 4.1 (CTFN fitting).
    Returns: (b1, b2, b3, k)
    """
    if rng is None:
        rng = np.random.default_rng()
    if u is None:
        u = rng.uniform()

    # P-prefix names clarify these are percentile values, not quartiles.
    pcts = np.percentile(port_rets, [3, 20, 50, 80, 97])
    P3, P20, P50, P80, P97 = pcts

    b1 = min(port_rets.min(), P3)
    b2 = P50
    b3 = P97

    EPS = 1e-12
    if b2 <= b1:
        b1 = b2 - EPS
    if b3 <= b2:
        b3 = b2 + EPS

    if u < 0.5:
        denom = (b2 - P20) / (b2 - b1)
    else:
        denom = (P80 - b2) / (b3 - b2)

    denom = np.clip(denom, EPS, 1 - EPS)
    k = np.log(0.5) / np.log(denom)
    k = np.clip(k, EPS, 100)

    return b1, b2, b3, k


def fit_citfn(port_rets, u=None, u_hat=None, rng=None):
    """Extended percentile-based fitting for Coherent ITFN.

    Steps I–IV:  same as CTFN  →  b1, b2, b3, k
    Step V:      b1_hat = min(R_min, Q1),  b3_hat = Q99
    Step VI:     k_hat from Q5 or Q95

    Returns: CoherentITFN instance
    """
    if rng is None:
        rng = np.random.default_rng()
    if u is None:
        u = rng.uniform()
    if u_hat is None:
        u_hat = rng.uniform()

    # P-prefix names clarify these are percentile values, not quartiles.
    # -- Steps I-IV (CTFN core) ------------------------------------------------
    pcts = np.percentile(port_rets, [1, 3, 5, 20, 50, 80, 95, 97, 99])
    P1, P3, P5, P20, P50, P80, P95, P97, P99 = pcts

    b1 = min(port_rets.min(), P3)
    b2 = P50
    b3 = P97

    EPS = 1e-12
    if b2 <= b1:
        b1 = b2 - EPS
    if b3 <= b2:
        b3 = b2 + EPS

    if u < 0.5:
        denom = (b2 - P20) / (b2 - b1)
    else:
        denom = (P80 - b2) / (b3 - b2)
    denom = np.clip(denom, EPS, 1 - EPS)
    k = np.log(0.5) / np.log(denom)
    k = np.clip(k, EPS, 100)

    # -- Step V: outer triangle endpoints --------------------------------------
    b1_hat = min(port_rets.min(), P1)
    b3_hat = P99

    # Ensure ordering constraints
    if b1_hat > b1:
        b1_hat = b1
    if b3_hat < b3:
        b3_hat = b3

    # Ensure non-degenerate outer triangle
    if b2 <= b1_hat:
        b1_hat = b2 - EPS
    if b3_hat <= b2:
        b3_hat = b2 + EPS

    # -- Step VI: non-membership shape parameter k_hat -------------------------
    alpha_hat = b2 - b1_hat
    beta_hat = b3_hat - b2

    if u_hat < 0.5:
        denom_hat = (b2 - P5) / alpha_hat if alpha_hat > EPS else 0.5
    else:
        denom_hat = (P95 - b2) / beta_hat if beta_hat > EPS else 0.5

    denom_hat = np.clip(denom_hat, EPS, 1 - EPS)
    k_hat = np.log(0.5) / np.log(denom_hat)
    k_hat = np.clip(k_hat, EPS, 100)

    return CoherentITFN(b1, b2, b3, k, b1_hat, b3_hat, k_hat)


# ==============================================================================
#  Section 5  MOGA OPERATORS  (extracted from credibilistic_moga.ipynb)
# ==============================================================================

# -- Problem parameters (Table 3 of the paper) --------------------------------
N = 18
KAPPA = 5
L = 0.08
U = 0.30
P_CVAR = 0.95
LAMBDA = 1.0
P_MUT = 5
K_MEDOIDS = 25
MIN_RETURN = 0.02

# -- MOGA hyperparameters -----------------------------------------------------
POP_SIZE = 180
G_MAX = 2000
R_MAX = 30


def repair(x, l=L, u=U):
    """Appendix B.4 — repair mechanism."""
    x = x.copy()
    active = x > 0
    if not active.any():
        return x
    B = x[active].sum()
    if abs(B - 1.0) < 1e-10:
        return x
    elif B > 1.0:
        num = x[active] - l
        denom = num.sum()
        if denom < 1e-12:
            x[active] = 1.0 / active.sum()
        else:
            x[active] = l + num / denom * (1 - l * active.sum())
    else:
        num = u - x[active]
        denom = num.sum()
        if denom < 1e-12:
            x[active] = 1.0 / active.sum()
        else:
            x[active] = u - num / denom * (u * active.sum() - 1)
    return x


def init_population(pop_size=POP_SIZE, n=N, kappa=KAPPA, l=L, u=U, rng=None):
    """Random feasible population."""
    if rng is None:
        rng = np.random.default_rng()
    pop = []
    while len(pop) < pop_size:
        x = np.zeros(n)
        active_idx = rng.choice(n, kappa, replace=False)
        weights = rng.uniform(l, u, kappa)
        x[active_idx] = weights
        x = repair(x, l, u)
        if abs(x.sum() - 1.0) < 1e-6 and (x[x > 0] >= l - 1e-9).all():
            pop.append(x)
    return np.array(pop)


def ccbex_crossover(x1, x2, l=L, u=U, lam=LAMBDA, rng=None):
    """Appendix B.1 — CCBEX crossover."""
    if rng is None:
        rng = np.random.default_rng()
    c1, c2 = x1.copy(), x2.copy()
    for i in range(len(x1)):
        if x1[i] > 0 and x2[i] > 0:
            if abs(x1[i] - x2[i]) < 1e-12:
                continue
            ri = rng.uniform()
            diff = abs(x1[i] - x2[i])
            _EPS_LOG = 1e-300  # guard against log(0)
            if ri <= 0.5:
                arg1 = max(np.exp(-(x1[i] - l) / (lam * diff))
                           + 2 * ri * (1 - np.exp(-(x1[i] - l) / (lam * diff))), _EPS_LOG)
                Cx1 = lam * np.log(arg1)
            else:
                arg1 = max(1 - (2 * ri - 1) * (1 - np.exp(-(u - x1[i]) / (lam * diff))), _EPS_LOG)
                Cx1 = -lam * np.log(arg1)
            if ri <= 0.5:
                arg2 = max(np.exp(-(x2[i] - l) / (lam * diff))
                           + 2 * ri * (1 - np.exp(-(x2[i] - l) / (lam * diff))), _EPS_LOG)
                Cx2 = lam * np.log(arg2)
            else:
                arg2 = max(1 - (2 * ri - 1) * (1 - np.exp(-(u - x2[i]) / (lam * diff))), _EPS_LOG)
                Cx2 = -lam * np.log(arg2)
            c1[i] = np.clip(x1[i] + Cx1 * diff, l, u)
            c2[i] = np.clip(x2[i] + Cx2 * diff, l, u)
    return c1, c2


def swap_mutation(x, l=L, u=U, rng=None):
    """Appendix B.2 — swap mutation."""
    if rng is None:
        rng = np.random.default_rng()
    x = x.copy()
    active = np.where(x > 0)[0]
    inactive = np.where(x == 0)[0]
    if len(active) == 0 or len(inactive) == 0:
        return x
    i = rng.choice(active)
    j = rng.choice(inactive)
    x[j] = x[i]  # transfer weight verbatim; keeps the weight in [l, u]
    x[i] = 0.0
    return x


def power_mutation(x, l=L, u=U, p=P_MUT, rng=None):
    """Appendix B.3 — power mutation."""
    if rng is None:
        rng = np.random.default_rng()
    x = x.copy()
    active = np.where(x > 0)[0]
    if len(active) == 0:
        return x
    i = rng.choice(active)
    rho_i = rng.uniform()
    sigma_i = rng.uniform()
    theta_i = (x[i] - l) / (u - l)
    chi_i = rho_i ** (1.0 / p)
    if theta_i < sigma_i:
        x[i] = x[i] - chi_i * (x[i] - l)
    else:
        x[i] = x[i] + chi_i * (u - x[i])
    x[i] = np.clip(x[i], l, u)
    return x


# ==============================================================================
#  Section 6  NSGA-II SELECTION
# ==============================================================================

def dominates(a, b):
    """True if objective-vector a dominates b (all minimised)."""
    return (np.all(a <= b)) and (np.any(a < b))


def fast_non_dominated_sort(obj):
    """NSGA-II non-dominated sort.  Returns list of fronts."""
    n = len(obj)
    dom_count = np.zeros(n, dtype=int)
    dom_set = [[] for _ in range(n)]
    fronts = [[]]

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if dominates(obj[i], obj[j]):
                dom_set[i].append(j)
            elif dominates(obj[j], obj[i]):
                dom_count[i] += 1
        if dom_count[i] == 0:
            fronts[0].append(i)

    k = 0
    while fronts[k]:
        next_front = []
        for i in fronts[k]:
            for j in dom_set[i]:
                dom_count[j] -= 1
                if dom_count[j] == 0:
                    next_front.append(j)
        fronts.append(next_front)
        k += 1

    return fronts[:-1]


def crowding_distance(obj, front):
    """Crowding distance for members of a front."""
    l = len(front)
    if l <= 2:
        return np.full(l, np.inf)
    dist = np.zeros(l)
    m = obj.shape[1]
    for o in range(m):
        vals = obj[front, o]
        order = np.argsort(vals)
        rng_o = vals[order[-1]] - vals[order[0]]
        if rng_o < 1e-12:
            continue
        dist[order[0]] = np.inf
        dist[order[-1]] = np.inf
        for i in range(1, l - 1):
            dist[order[i]] += (vals[order[i + 1]] - vals[order[i - 1]]) / rng_o
    return dist


def nsga2_select(pop, obj_combined, pop_size):
    """Select pop_size solutions from combined parent+offspring pool."""
    fronts = fast_non_dominated_sort(obj_combined)
    chosen = []
    for front in fronts:
        if len(chosen) + len(front) <= pop_size:
            chosen.extend(front)
        else:
            needed = pop_size - len(chosen)
            cd = crowding_distance(obj_combined, front)
            order = np.argsort(-cd)
            chosen.extend([front[i] for i in order[:needed]])
            break
    chosen = np.array(chosen[:pop_size])
    return pop[chosen], obj_combined[chosen]


# ==============================================================================
#  Section 7  K-MEDOIDS CLUSTERING
# ==============================================================================

def kmedoids_representative(pareto_solutions, pareto_obj, k=K_MEDOIDS, rng=None):
    """Select K representative solutions using KMeans-medoid approximation."""
    m = len(pareto_solutions)
    if m <= k:
        return pareto_solutions, pareto_obj

    seed = int(rng.integers(1_000_000)) if rng is not None else 42
    km = KMeans(n_clusters=k, random_state=seed, n_init='auto', max_iter=500)
    km.fit(pareto_solutions)

    chosen = []
    dists_to_centers = np.sum(
        (pareto_solutions[:, None, :] - km.cluster_centers_[None, :, :]) ** 2,
        axis=-1
    )

    for ci in range(k):
        d = dists_to_centers[:, ci].copy()
        for already in chosen:
            d[already] = np.inf
        chosen.append(int(np.argmin(d)))

    chosen = np.array(chosen)
    return pareto_solutions[chosen], pareto_obj[chosen]


# ==============================================================================
#  Section 8  OBJECTIVE COMPUTATION & MOGA LOOP
# ==============================================================================

# -- Risk measure keys ---------------------------------------------------------
RISK_KEY = {
    'I': 'sv', 'II': 'masd', 'III': 'cvar', 'IV': 'evar',
    'I_ITFN': 'sv',
}

# -- Which models use ITFN? ----------------------------------------------------
_ITFN_MODELS = {'I_ITFN'}


def compute_evar_empirical(returns, beta=0.95):
    """Empirical EVaR from a return series using the MGF definition.

    EVaR_beta(X) = inf_{t>0} t^{-1} log(M_X(t) / (1-beta)).
    The implementation minimises over positive t using scipy when available.
    """
    losses = -np.asarray(returns, dtype=float)
    if losses.size == 0:
        return np.nan
    if _HAS_SCIPY_OPT:
        log_one_minus_beta = np.log(1.0 - beta)

        def obj(t):
            z = t * losses
            z_max = z.max()
            log_mgf = z_max + np.log(np.mean(np.exp(z - z_max)))
            return (log_mgf - log_one_minus_beta) / t

        res = _minimize_scalar(obj, bounds=(1e-4, 50.0), method='bounded',
                               options={'xatol': 1e-6})
        return float(res.fun)

    # Fallback: a simple tail-based proxy when scipy is not available.
    return float(np.quantile(losses, beta))


def compute_objectives_ctfn(weights, z, R_matrix, p_cvar=P_CVAR, u=None, rng=None):
    """Return CTFN objectives dict for a portfolio."""
    R = portfolio_returns(weights, z, R_matrix)
    b1, b2, b3, k = fit_ctfn(R, u=u, rng=rng)
    e = ctfn_mean(b1, b2, b3, k)
    return {
        'b1': b1, 'b2': b2, 'b3': b3, 'k': k,
        'mean': e,
        'sv': ctfn_sv(b1, b2, b3, k, e),
        'masd': ctfn_masd(b1, b2, b3, k),
        'cvar': ctfn_cvar(b1, b2, b3, k, p_cvar),
        'evar': compute_evar_empirical(R, 0.95),
        'skewness': ctfn_skewness(b1, b2, b3, k, e),
        'semikurt': ctfn_sk(b1, b2, b3, k, e),
    }


def compute_objectives_evar(weights, z, R_matrix, p_cvar=P_CVAR, p_evar=0.95,
                            u=None, rng=None):
    """Return CTFN objectives dict with EVaR replacing CVaR."""
    R = portfolio_returns(weights, z, R_matrix)
    b1, b2, b3, k = fit_ctfn(R, u=u, rng=rng)
    e = ctfn_mean(b1, b2, b3, k)
    return {
        'b1': b1, 'b2': b2, 'b3': b3, 'k': k,
        'mean': e,
        'sv': ctfn_sv(b1, b2, b3, k, e),
        'masd': ctfn_masd(b1, b2, b3, k),
        'cvar': ctfn_cvar(b1, b2, b3, k, p_cvar),
        'evar': compute_evar_empirical(R, p_evar),
        'skewness': ctfn_skewness(b1, b2, b3, k, e),
        'semikurt': ctfn_sk(b1, b2, b3, k, e),
    }


def compute_objectives_itfn(weights, z, R_matrix, u=None, u_hat=None, rng=None):
    """Return ITFN objectives dict for a portfolio."""
    R = portfolio_returns(weights, z, R_matrix)
    citfn = fit_citfn(R, u=u, u_hat=u_hat, rng=rng)
    e = citfn.mean()
    return {
        'b1': citfn.b1, 'b2': citfn.b2, 'b3': citfn.b3, 'k': citfn.k,
        'b1_hat': citfn.b1_hat, 'b3_hat': citfn.b3_hat, 'k_hat': citfn.k_hat,
        'delta_L': citfn.delta_L, 'delta_R': citfn.delta_R,
        'mean': e,
        'sv': citfn.semivariance(e),
        'skewness': citfn.skewness(e),
        'semikurt': citfn.semikurtosis(e),
        'total_hesitation': citfn.total_hesitation(),
        'hesitation_asymmetry': citfn.hesitation_asymmetry(),
        '_citfn': citfn,  # keep reference for diagnostics
    }


def objectives_from_obs(obs, model):
    """Convert objectives dict → minimisation vector for NSGA-II.
    f = [-mean, risk, -skewness, semikurt]  — all minimised.
    """
    risk = RISK_KEY[model]
    return np.array([-obs['mean'], obs[risk], -obs['skewness'], obs['semikurt']])


def eval_population(pop, R_matrix, model, p_cvar=P_CVAR,
                    u_fixed=None, u_hat_fixed=None, rng=None):
    """Evaluate all solutions in the population.
    Returns (obj_matrix, raw_obs_list).
    """
    is_itfn = model in _ITFN_MODELS
    obj_list = []
    obs_list = []
    for x in pop:
        z = (x > 0).astype(float)
        if model == 'IV':
            obs = compute_objectives_evar(x, z, R_matrix, p_cvar=p_cvar,
                                          p_evar=0.95, u=u_fixed, rng=rng)
        elif is_itfn:
            obs = compute_objectives_itfn(x, z, R_matrix,
                                          u=u_fixed, u_hat=u_hat_fixed, rng=rng)
        else:
            obs = compute_objectives_ctfn(x, z, R_matrix, p_cvar=p_cvar,
                                          u=u_fixed, rng=rng)
        obj_list.append(objectives_from_obs(obs, model))
        obs_list.append(obs)
    return np.array(obj_list), obs_list


def run_moga(R_matrix, model, pop_size=POP_SIZE, g_max=G_MAX,
             n=N, kappa=KAPPA, l=L, u=U, lam=LAMBDA, p_mut=P_MUT,
             cp=0.7, mp=0.3, p_cvar=P_CVAR,
             verbose=True, seed=None):
    """Single MOGA run.  Returns Pareto-optimal (solutions, objectives)."""
    rng = np.random.default_rng(seed)
    u_fixed = rng.uniform()
    u_hat_fixed = rng.uniform()  # for ITFN shape consistency

    pop = init_population(pop_size, n, kappa, l, u, rng)
    obj, _ = eval_population(pop, R_matrix, model, p_cvar, u_fixed, u_hat_fixed, rng)

    for g in range(g_max):
        offspring = []
        n_pop = len(pop)

        while len(offspring) < n_pop:
            i1, i2 = rng.integers(0, n_pop, 2)
            p1, p2 = pop[i1], pop[i2]

            if rng.uniform() < cp:
                c1, c2 = ccbex_crossover(p1, p2, l, u, lam, rng)
            else:
                c1, c2 = p1.copy(), p2.copy()

            if rng.uniform() < mp:
                c1 = swap_mutation(c1, l, u, rng)
            if rng.uniform() < mp:
                c2 = swap_mutation(c2, l, u, rng)

            if rng.uniform() < mp:
                c1 = power_mutation(c1, l, u, p_mut, rng)
            if rng.uniform() < mp:
                c2 = power_mutation(c2, l, u, p_mut, rng)

            c1 = repair(c1, l, u)
            c2 = repair(c2, l, u)

            for c in [c1, c2]:
                active_idx = np.where(c > 0)[0]
                if len(active_idx) != kappa:
                    continue
            offspring.extend([c1, c2])

        offspring = np.array(offspring[:n_pop])
        off_obj, _ = eval_population(offspring, R_matrix, model, p_cvar,
                                     u_fixed, u_hat_fixed, rng)

        combined = np.vstack([pop, offspring])
        combined_obj = np.vstack([obj, off_obj])
        pop, obj = nsga2_select(combined, combined_obj, n_pop)

        if verbose and (g + 1) % max(1, g_max // 5) == 0:
            fronts = fast_non_dominated_sort(obj)
            n_pf = len(fronts[0]) if fronts else 0
            print(f'  Gen {g + 1:4d}/{g_max} | Pareto front size: {n_pf}')
            sys.stdout.flush()

    fronts = fast_non_dominated_sort(obj)
    pf_idx = fronts[0] if fronts else list(range(len(pop)))
    return pop[pf_idx], obj[pf_idx]


def optimise_model(R_matrix, model, r_max=R_MAX, pop_size=POP_SIZE, g_max=G_MAX,
                   cp=0.7, mp=0.3, min_return=MIN_RETURN, k_med=K_MEDOIDS,
                   verbose=True, base_seed=0, p_cvar=P_CVAR):
    """Multi-run optimisation with pooling, filtration, and clustering.

    Returns
    -------
    rep_sol : (k_med, n)   representative portfolio weight vectors
    rep_obj : (k_med, 4)   their objective values (minimisation convention)
    rep_raw : list[dict]   full observations per rep solution
    all_pf  : (M, n)       all pooled Pareto solutions before clustering
    """
    rng = np.random.default_rng(base_seed)
    all_sol = []
    all_obj_l = []

    for run in range(r_max):
        seed_r = rng.integers(1_000_000)
        if verbose:
            print(f'[Model {model}]  Run {run + 1}/{r_max}  (seed={seed_r})')
            sys.stdout.flush()
        pf_sol, pf_obj = run_moga(
            R_matrix, model,
            pop_size=pop_size, g_max=g_max,
            cp=cp, mp=mp, p_cvar=p_cvar,
            verbose=verbose, seed=int(seed_r)
        )
        all_sol.append(pf_sol)
        all_obj_l.append(pf_obj)

    all_sol = np.vstack(all_sol)
    all_obj = np.vstack(all_obj_l)

    # Re-extract Pareto front from pooled results
    fronts = fast_non_dominated_sort(all_obj)
    pf_idx = fronts[0] if fronts else list(range(len(all_sol)))
    pf_sol = all_sol[pf_idx]
    pf_obj = all_obj[pf_idx]

    if verbose:
        print(f'  Pooled Pareto front: {len(pf_sol)} solutions')
        sys.stdout.flush()

    # -- Filtration ------------------------------------------------------------
    mask = (-pf_obj[:, 0] >= min_return) & (-pf_obj[:, 2] >= 0)
    if mask.sum() == 0:
        if verbose:
            print('  [Warning] No solutions passed filtration — relaxing to top-50% mean')
        threshold = np.percentile(-pf_obj[:, 0], 50)
        mask = (-pf_obj[:, 0] >= threshold) & (-pf_obj[:, 2] >= 0)
    if mask.sum() == 0:
        mask = np.ones(len(pf_sol), dtype=bool)

    fs_sol = pf_sol[mask]
    fs_obj = pf_obj[mask]

    if verbose:
        print(f'  After filtration: {len(fs_sol)} solutions')
        sys.stdout.flush()

    # -- K-Medoids -------------------------------------------------------------
    rep_sol, rep_obj = kmedoids_representative(
        fs_sol, fs_obj, k=min(k_med, len(fs_sol)), rng=rng)

    # Re-evaluate representative solutions for full observations
    is_itfn = model in _ITFN_MODELS
    rep_raw = []
    u_fixed = rng.uniform()
    u_hat_fixed = rng.uniform()
    for x in rep_sol:
        z = (x > 0).astype(float)
        if model == 'IV':
            obs = compute_objectives_evar(x, z, R_matrix, p_cvar=p_cvar,
                                          p_evar=0.95, u=u_fixed, rng=rng)
        elif is_itfn:
            obs = compute_objectives_itfn(x, z, R_matrix,
                                          u=u_fixed, u_hat=u_hat_fixed, rng=rng)
        else:
            obs = compute_objectives_ctfn(x, z, R_matrix, u=u_fixed, rng=rng)
        rep_raw.append(obs)

    if verbose:
        print(f'  Representative solutions: {len(rep_sol)}')
        sys.stdout.flush()

    return rep_sol, rep_obj, rep_raw, pf_sol


# ==============================================================================
#  Section 9  ITFN vs EVaR COMPARISON RUNNER
# ============================================================================== 


def _load_market_data(market, data_dir='data'):
    market = market.lower()
    train_path = f'{data_dir}/{market}_train.csv'
    test_path = f'{data_dir}/{market}_test.csv'
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    return train, test


def _portfolio_test_metrics(weights, z, R_test):
    """Compute simple out-of-sample summary metrics for a portfolio.

    Returns
    -------
    cum_return : float  — total compounded return over the test window
    sharpe     : float  — annualised Sharpe ratio (assumes monthly returns,
                          multiplied by sqrt(12); no risk-free rate)
    max_dd     : float  — maximum drawdown (positive value, e.g. 0.15 = 15%)

    Note: R_test (and R_train) store monthly returns as percentage points
    (e.g. 1.5 means +1.5%), matching the CTFN fitting convention where b2=Q50
    yields values like 1.5 (not 0.015).  Division by 100 converts to decimal
    before compounding.
    """
    ret = portfolio_returns(weights, z, R_test)
    ret_dec = np.asarray(ret, dtype=float) / 100.0  # data is in percentage points — convert to decimal
    if len(ret_dec) == 0:
        return {'cum_return': np.nan, 'sharpe': np.nan, 'max_dd': np.nan}
    cum = np.cumprod(1.0 + ret_dec) - 1.0
    final_cum = cum[-1]
    std = ret_dec.std(ddof=0)
    # Annualised Sharpe: mean/std * sqrt(12) for monthly data (standard convention)
    sharpe = (ret_dec.mean() / std) * np.sqrt(12) if std > 1e-12 else np.nan
    peak = np.maximum.accumulate(1.0 + cum)
    drawdown = 1.0 + cum - peak
    max_dd = -np.min(drawdown)
    return {'cum_return': final_cum, 'sharpe': sharpe, 'max_dd': max_dd}


def _select_best_representation(rep_sol, rep_raw, model):
    """Choose the representative portfolio that looks best under a simple utility score."""
    if len(rep_sol) == 0:
        return None, None
    if len(rep_sol) == 1:
        return rep_sol[0], rep_raw[0]

    scores = []
    for obs in rep_raw:
        risk = obs.get('evar', obs.get('sv', 0.0))
        utility = obs['mean'] - 0.5 * risk + 0.25 * obs['skewness'] - 0.1 * obs['semikurt']
        scores.append(utility)
    idx = int(np.argmax(scores))
    return rep_sol[idx], rep_raw[idx]


def run_itfn_vs_evar_comparison(markets=('nse', 'nyse'), quick_test=True, verbose=True,
                                 save_plots=True):
    """Run a lightweight ITFN-vs-EVaR comparison and print the final summary."""
    if quick_test:
        pop_size, g_max, r_max = 30, 50, 2
    else:
        # Substantive run — enough for convergence (~30-45 min total).
        # Paper params (180/2000/30) take 30+ hours; use for final publication.
        pop_size, g_max, r_max = 80, 250, 5

    if verbose:
        mode = 'QUICK TEST' if quick_test else 'FULL'
        print(f'\n{"="*70}')
        print(f'  ITFN vs EVaR Comparison  ({mode})')
        print(f'  pop_size={pop_size}  g_max={g_max}  r_max={r_max}')
        print(f'{"="*70}')
        sys.stdout.flush()

    rows = []
    for market in markets:
        train, test = _load_market_data(market)
        R_train = train.drop(columns=['Date']).to_numpy(dtype=float)
        R_test = test.drop(columns=['Date']).to_numpy(dtype=float)

        if verbose:
            print(f'\n{"-"*70}')
            print(f'  Market: {market.upper()}  (train={R_train.shape[0]} obs, test={R_test.shape[0]} obs, {R_train.shape[1]} assets)')
            print(f'{"-"*70}')
            sys.stdout.flush()

        # ITFN model I
        if verbose:
            print(f'\n  >> Running ITFN Model I ({r_max} runs x {g_max} gen)...')
            sys.stdout.flush()
        t0 = time.time()
        rep_sol_itfn, rep_obj_itfn, rep_raw_itfn, _ = optimise_model(
            R_train, 'I_ITFN', r_max=r_max, pop_size=pop_size, g_max=g_max,
            verbose=verbose, base_seed=100 + (0 if market == 'nse' else 1)
        )
        best_sol_itfn, best_obs_itfn = _select_best_representation(rep_sol_itfn, rep_raw_itfn, 'I_ITFN')
        if verbose:
            print(f'  >> ITFN done in {time.time()-t0:.1f}s  ({len(rep_sol_itfn)} representatives)')
            sys.stdout.flush()

        # EVaR model IV
        if verbose:
            print(f'\n  >> Running EVaR Model IV ({r_max} runs x {g_max} gen)...')
            sys.stdout.flush()
        t0 = time.time()
        rep_sol_evar, rep_obj_evar, rep_raw_evar, _ = optimise_model(
            R_train, 'IV', r_max=r_max, pop_size=pop_size, g_max=g_max,
            verbose=verbose, base_seed=200 + (0 if market == 'nse' else 1)
        )
        best_sol_evar, best_obs_evar = _select_best_representation(rep_sol_evar, rep_raw_evar, 'IV')
        if verbose:
            print(f'  >> EVaR done in {time.time()-t0:.1f}s  ({len(rep_sol_evar)} representatives)')
            sys.stdout.flush()

        metrics_itfn = _portfolio_test_metrics(best_sol_itfn, (best_sol_itfn > 0).astype(float), R_test)
        metrics_evar = _portfolio_test_metrics(best_sol_evar, (best_sol_evar > 0).astype(float), R_test)

        rows.append({
            'Market': market.upper(),
            'Model': 'ITFN Model I',
            'Mean(train)': best_obs_itfn['mean'],
            'Risk(train)': best_obs_itfn.get('sv', np.nan),
            'Skew(train)': best_obs_itfn['skewness'],
            'SemiKurt(train)': best_obs_itfn['semikurt'],
            'CumReturn(test)': metrics_itfn['cum_return'],
            'Sharpe(test)': metrics_itfn['sharpe'],
            'MaxDD(test)': metrics_itfn['max_dd'],
        })
        rows.append({
            'Market': market.upper(),
            'Model': 'EVaR Model IV',
            'Mean(train)': best_obs_evar['mean'],
            'Risk(train)': best_obs_evar.get('evar', np.nan),
            'Skew(train)': best_obs_evar['skewness'],
            'SemiKurt(train)': best_obs_evar['semikurt'],
            'CumReturn(test)': metrics_evar['cum_return'],
            'Sharpe(test)': metrics_evar['sharpe'],
            'MaxDD(test)': metrics_evar['max_dd'],
        })

        if verbose:
            print(f'\n  Results for {market.upper()}:')
            print(f'    ITFN Model I  ->  mean={best_obs_itfn["mean"]:.4f}  sv={best_obs_itfn.get("sv", np.nan):.4f}'
                  f'  skew={best_obs_itfn["skewness"]:.4f}  cum={metrics_itfn["cum_return"]:.4f}'
                  f'  sharpe={metrics_itfn["sharpe"]:.4f}  max_dd={metrics_itfn["max_dd"]:.4f}')
            print(f'    EVaR Model IV ->  mean={best_obs_evar["mean"]:.4f}  evar={best_obs_evar.get("evar", np.nan):.4f}'
                  f'  skew={best_obs_evar["skewness"]:.4f}  cum={metrics_evar["cum_return"]:.4f}'
                  f'  sharpe={metrics_evar["sharpe"]:.4f}  max_dd={metrics_evar["max_dd"]:.4f}')
            sys.stdout.flush()

    summary = pd.DataFrame(rows)
    if verbose:
        print(f'\n{"="*70}')
        print('  FINAL COMPARISON SUMMARY')
        print(f'{"="*70}')
        # Format the dataframe for nicer printing
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', 120)
        pd.set_option('display.float_format', lambda x: f'{x:.6f}')
        print(summary.to_string(index=False))
        pd.reset_option('display.float_format')
        print(f'{"="*70}')
        sys.stdout.flush()

    return summary


# ==============================================================================
#  Section 10  SANITY CHECKS

def run_itfn_sanity_checks(verbose=True):
    """Validate ITFN implementation.

    Tests:
    1. CTFN degeneracy (k=1) -- zero hesitation reduces to CTFN moments
    2. Self-duality -- Cr^I{>=t} + Cr^I{<t} = 1
    3. Constraint satisfaction -- mu(t) + nu(t) <= 1 for well-formed ITFN
    4. k=1 CTFN closed form
    5. ITFN moments differ from CTFN when hesitation > 0
    6. Hesitation diagnostics
    7. Numerical CTFN cross-check via quad
    """
    all_pass = True
    n_tests = 0

    def check(name, condition, detail=""):
        nonlocal all_pass, n_tests
        n_tests += 1
        if condition:
            if verbose:
                print(f'  [PASS] {name}')
        else:
            all_pass = False
            if verbose:
                print(f'  [FAIL] {name}  {detail}')

    if verbose:
        print('=' * 70)
        print('  ITFN Sanity Checks')
        print('=' * 70)

    # -- Test 1: CTFN degeneracy with k=1 --------------------------------------
    # At k=1, k_hat=1, b1_hat=b1, b3_hat=b3:
    #   mu(t) = (t-b1)/alpha  on [b1,b2]  and  nu(t) = (b2-t)/alpha  on [b1,b2]
    #   so mu + nu = 1  =>  s(t) = mu(t)  =>  ITFN credibility == CTFN credibility
    if verbose:
        print('\n-- Test 1: CTFN degeneracy (k=1, zero hesitation -> CTFN moments) --')

    b1, b2, b3, k = -0.05, 0.01, 0.15, 1.0
    citfn_deg = CoherentITFN(b1, b2, b3, k, b1, b3, k)

    e_ctfn = ctfn_mean(b1, b2, b3, k)
    e_itfn = citfn_deg.mean()
    check('Mean degeneracy (k=1)', abs(e_itfn - e_ctfn) < 1e-8,
          f'CTFN={e_ctfn:.10f}  ITFN={e_itfn:.10f}  diff={abs(e_itfn-e_ctfn):.2e}')

    sv_ctfn = ctfn_sv(b1, b2, b3, k, e_ctfn)
    sv_itfn = citfn_deg.semivariance(e_itfn)
    check('Semivariance degeneracy (k=1)', abs(sv_itfn - sv_ctfn) < 1e-8,
          f'CTFN={sv_ctfn:.10e}  ITFN={sv_itfn:.10e}  diff={abs(sv_itfn-sv_ctfn):.2e}')

    sk_ctfn = ctfn_skewness(b1, b2, b3, k, e_ctfn)
    sk_itfn = citfn_deg.skewness(e_itfn)
    check('Skewness degeneracy (k=1)', abs(sk_itfn - sk_ctfn) < 1e-8,
          f'CTFN={sk_ctfn:.10e}  ITFN={sk_itfn:.10e}  diff={abs(sk_itfn-sk_ctfn):.2e}')

    skt_ctfn = ctfn_sk(b1, b2, b3, k, e_ctfn)
    skt_itfn = citfn_deg.semikurtosis(e_itfn)
    check('Semikurtosis degeneracy (k=1)', abs(skt_itfn - skt_ctfn) < 1e-8,
          f'CTFN={skt_ctfn:.10e}  ITFN={skt_itfn:.10e}  diff={abs(skt_itfn-skt_ctfn):.2e}')

    # Verify mu + nu = 1 for the degenerate k=1 case
    ts_deg = np.linspace(b1, b3, 100)
    max_dev = max(abs(citfn_deg.membership(t) + citfn_deg.nonmembership(t) - 1.0) for t in ts_deg)
    check('mu+nu=1 for degenerate k=1 ITFN', max_dev < 1e-12,
          f'max |mu+nu-1| = {max_dev:.2e}')

    # -- Test 2: Self-duality --------------------------------------------------
    if verbose:
        print('\n-- Test 2: Self-duality  Cr^I{>=t} + Cr^I{<t} = 1 --')

    # Use wider outer triangle to ensure mu+nu <= 1
    citfn_test = CoherentITFN(-0.05, 0.01, 0.12, 0.7, -0.12, 0.20, 0.5)
    ts = np.linspace(-0.15, 0.25, 100)
    max_violation = 0.0
    for t in ts:
        total = citfn_test.credibility_geq(t) + citfn_test.credibility_leq(t)
        max_violation = max(max_violation, abs(total - 1.0))
    check('Self-duality (100 points)', max_violation < 1e-12,
          f'max violation = {max_violation:.2e}')

    # -- Test 3: Constraint satisfaction mu+nu <= 1 ----------------------------
    # nonmembership() now clamps nu(t) to (1 - mu(t)) so the IFS axiom
    # mu(t) + nu(t) <= 1 is guaranteed at every point for all k.
    if verbose:
        print('\n-- Test 3: mu(t) + nu(t) <= 1 (IFS axiom, all k) --')

    # Use k=1, k_hat=1 with wider outer triangle -> mu+nu <= 1 guaranteed
    citfn_valid = CoherentITFN(-0.05, 0.01, 0.12, 1.0, -0.12, 0.20, 1.0)
    ts3 = np.linspace(citfn_valid.b1_hat, citfn_valid.b3_hat, 1000)
    max_sum = 0.0
    for t in ts3:
        s = citfn_valid.membership(t) + citfn_valid.nonmembership(t)
        max_sum = max(max_sum, s)
    check('mu + nu <= 1 (k=1,k_hat=1, wide outer, 1000 pts)', max_sum <= 1.0 + 1e-12,
          f'max sum = {max_sum:.10f}')

    # k < 1 case: without clamping, raw mu + raw_nu could exceed 1.
    # nonmembership() clamps nu so the axiom is enforced regardless of k.
    citfn_k07 = CoherentITFN(-0.05, 0.01, 0.12, 0.7, -0.20, 0.30, 0.7)
    ts3b = np.linspace(citfn_k07.b1_hat, citfn_k07.b3_hat, 1000)
    max_sum_k07 = max(citfn_k07.membership(t) + citfn_k07.nonmembership(t) for t in ts3b)
    check('mu + nu <= 1 (k=0.7, clamped, 1000 pts)', max_sum_k07 <= 1.0 + 1e-12,
          f'max sum = {max_sum_k07:.10f}')
    max_score = max(citfn_k07.score_membership(t) for t in ts3b)
    min_score = min(citfn_k07.score_membership(t) for t in ts3b)
    check('score s(t) in [0,1] (k=0.7, 1000 pts)', min_score >= -1e-12 and max_score <= 1.0 + 1e-12,
          f'min={min_score:.10f}  max={max_score:.10f}')

    # -- Test 4: k=1 CTFN closed form -----------------------------------------
    if verbose:
        print('\n-- Test 4: k=1 CTFN closed form --')

    b1_t, b2_t, b3_t, k_t = -0.05, 0.01, 0.15, 1.0
    e_formula = ctfn_mean(b1_t, b2_t, b3_t, k_t)
    e_closed = (b1_t + 2 * b2_t + b3_t) / 4
    check('k=1 Mean', abs(e_formula - e_closed) < 1e-10,
          f'formula={e_formula:.10f}  closed={e_closed:.10f}')

    s_formula = ctfn_skewness(b1_t, b2_t, b3_t, k_t)
    alpha_t = b2_t - b1_t
    beta_t = b3_t - b2_t
    s_closed = (beta_t + alpha_t) ** 2 * (beta_t - alpha_t) / 32
    check('k=1 Skewness', abs(s_formula - s_closed) < 1e-12,
          f'formula={s_formula:.10e}  closed={s_closed:.10e}')

    # -- Test 5: ITFN moments differ from CTFN when hesitation > 0 -----------
    if verbose:
        print('\n-- Test 5: ITFN moments != CTFN when hesitation > 0 --')

    # Use ASYMMETRIC hesitation: dL=0.05, dR=0.04 so mean correction != 0
    citfn_nondeg = CoherentITFN(-0.05, 0.01, 0.15, 1.0, -0.10, 0.19, 1.0)
    e_nd = citfn_nondeg.mean()
    e_ctfn_nd = ctfn_mean(-0.05, 0.01, 0.15, 1.0)
    check('Mean differs with hesitation', abs(e_nd - e_ctfn_nd) > 1e-6,
          f'ITFN={e_nd:.8f}  CTFN={e_ctfn_nd:.8f}')

    sv_nd = citfn_nondeg.semivariance(e_nd)
    sv_ctfn_nd = ctfn_sv(-0.05, 0.01, 0.15, 1.0, e_nd)
    check('Semivariance differs with hesitation', abs(sv_nd - sv_ctfn_nd) > 1e-8,
          f'ITFN={sv_nd:.8e}  CTFN={sv_ctfn_nd:.8e}')

    # -- Test 6: Hesitation diagnostics ----------------------------------------
    if verbose:
        print('\n-- Test 6: Hesitation diagnostics --')

    th = citfn_nondeg.total_hesitation()
    check('Total hesitation > 0', th > 0, f'total_hesitation = {th:.6f}')

    ha = citfn_nondeg.hesitation_asymmetry()
    check('Hesitation asymmetry in [-1, 1]', -1.0 <= ha <= 1.0,
          f'asymmetry = {ha:.6f}')

    # Zero hesitation case
    th_zero = citfn_deg.total_hesitation()
    check('Zero hesitation for degenerate CITFN', th_zero < 1e-8,
          f'total_hesitation = {th_zero:.10f}')

    # -- Test 7: Numerical CTFN mean cross-check via quad ----------------------
    if verbose:
        print('\n-- Test 7: Numerical CTFN mean via quad --')

    if _HAS_SCIPY:
        b1_q, b2_q, b3_q, k_q = -0.08, 0.005, 0.10, 0.65
        citfn_q = CoherentITFN(b1_q, b2_q, b3_q, k_q, b1_q, b3_q, k_q)
        # For k != 1, mu+nu != 1 in general, so ITFN != CTFN
        # But we can verify the numerical integration is self-consistent:
        e_q = citfn_q.mean()
        # Recompute via direct integration of credibility_geq
        pos, _ = _quad(citfn_q.credibility_geq, 0, b3_q + 0.01, limit=200,
                       points=[b1_q, b2_q, b3_q])
        neg, _ = _quad(citfn_q.credibility_leq, b1_q - 0.01, 0, limit=200,
                       points=[b1_q, b2_q, b3_q])
        e_direct = pos - neg
        check('Numerical mean self-consistency', abs(e_q - e_direct) < 1e-10,
              f'mean()={e_q:.10f}  direct={e_direct:.10f}  diff={abs(e_q-e_direct):.2e}')
    else:
        if verbose:
            print('  [SKIP] scipy not available')

    # -- Summary ---------------------------------------------------------------
    if verbose:
        print('\n' + '=' * 70)
        status = 'ALL PASSED' if all_pass else 'SOME FAILED'
        print(f'  {n_tests} tests run -- {status}')
        print('=' * 70)

    return all_pass


# ==============================================================================
#  Section 11  MODULE ENTRY POINT
# ============================================================================== 

if __name__ == '__main__':
    run_itfn_sanity_checks(verbose=True)
    print('\nRunning ITFN vs EVaR comparison (quick test)...')
    run_itfn_vs_evar_comparison(quick_test=True, verbose=True)
