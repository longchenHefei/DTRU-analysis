#!/usr/bin/env python3
"""Failure-boosting cost for DTRU-2048.

Reproduces the decryption-failure model of DTRU (tricyclotomic ring,
Double-E8 decoder, ciphertext compression) and estimates the classical
and quantum cost of one decryption failure under failure boosting
(D'Anvers–Vercauteren–Verbauwhede, PKC 2019 / ePrint 2018/1089).

The search is single-target: DTRU hashes ID(pk) into the coins, so a weak
ciphertext does not transfer across keys.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import differential_evolution, minimize

Q = 3457
Q2 = 1024
N = 2048
HALF = N // 2
T_THRESH = (Q - 1) ** 2 / 2  # (q-1)^2 / 2
N_BLOCKS = N // 16

# CBD parameters from Table 1: (k_g, k_f') = (B5, B1), (k_r, k_e) = (B2, B1)
ETA_G, ETA_F, ETA_R, ETA_E = 5, 1, 2, 1


def cbd_pmf(eta: int):
    """Exact law of CBD(eta) = sum_{i<eta} (a_i - b_i), a_i,b_i uniform bits."""
    counts = {}
    total = 1 << (2 * eta)
    for mask in range(total):
        x = 0
        for i in range(eta):
            x += ((mask >> i) & 1) - ((mask >> (eta + i)) & 1)
        counts[x] = counts.get(x, 0) + 1
    xs = sorted(counts)
    p = np.array([counts[x] / total for x in xs], dtype=np.float64)
    return np.array(xs, dtype=np.float64), p


def rounding_pmf(q: int, q2: int):
    """Law of decompress(compress(x)) - x for x uniform in Z_q.

    Rounding is floor(y + 1/2), matching the spec's nearest-integer brackets.
    """
    counts = {}
    for x in range(q):
        c = (2 * q2 * x + q) // (2 * q)
        c %= q2
        xp = (2 * q * c + q2) // (2 * q2)
        xp %= q
        e = xp - x
        if e > q // 2:
            e -= q
        if e < -(q // 2):
            e += q
        counts[e] = counts.get(e, 0) + 1
    xs = sorted(counts)
    p = np.array([counts[x] / q for x in xs], dtype=np.float64)
    return np.array(xs, dtype=np.float64), p


def moments(xs, ps):
    mu = float(np.dot(ps, xs))
    sec = float(np.dot(ps, xs * xs))
    return mu, sec


def kl_bits(q, p):
    s = 0.0
    for qi, pi in zip(q, p):
        if qi <= 0.0:
            continue
        s += qi * math.log2(qi / pi)
    return s


def tilt_pmf(xs, ps, alpha, beta):
    """Exponential tilt q(x) ∝ p(x) exp(alpha x + beta x^2)."""
    logw = np.log(ps) + alpha * xs + beta * (xs * xs)
    logw -= logw.max()
    w = np.exp(logw)
    w /= w.sum()
    return w


def e8_codewords():
    H = np.array(
        [
            [1, 1, 1, 1, 0, 0, 0, 0],
            [0, 0, 1, 1, 1, 1, 0, 0],
            [0, 0, 0, 0, 1, 1, 1, 1],
            [0, 1, 0, 1, 0, 1, 0, 1],
        ],
        dtype=np.int64,
    )
    codes = []
    for m in range(16):
        bits = np.array([(m >> i) & 1 for i in range(4)], dtype=np.int64)
        codes.append((bits @ H) & 1)
    return codes


def code_tilt(codes, gamma):
    wts = np.array([c.sum() for c in codes], dtype=np.float64)
    logw = gamma * wts
    logw -= logw.max()
    pr = np.exp(logw)
    pr /= pr.sum()
    return pr, wts


def w_moments_from_code(codes, pr):
    """Average bit mean / second moment. w in {0,1}, so both equal the bit bias."""
    acc = np.zeros(8, dtype=np.float64)
    for c, p in zip(codes, pr):
        acc += p * c
    # repetition does not change the marginal of a uniformly indexed coefficient
    pi = float(acc.mean())
    return pi, pi, acc


def mul_by_x(a, half):
    out = np.empty_like(a)
    last = a[-1]
    out[0] = -last
    out[1:] = a[:-1]
    out[half] += last
    return out


def all_Q(r):
    """Q[k] = sum_j (r * x^j)_k^2 for the tricyclotomic ring."""
    n = r.shape[0]
    half = n // 2
    acc = np.array(r, dtype=np.float64).copy()
    Qacc = np.zeros(n, dtype=np.float64)
    for _ in range(n):
        Qacc += acc * acc
        acc = mul_by_x(acc, half)
    return Qacc


def ring_factors(n):
    half = n // 2
    fac = np.empty(n, dtype=np.float64)
    for k in range(half):
        fac[k] = 1.5 * n - 1 - k
    fac[half:] = 1.5 * n
    return fac


def block_indices(n):
    half = n // 2
    blocks = []
    for i in range(n // 16):
        a = np.arange(8 * i, 8 * i + 8)
        blocks.append(np.concatenate([a, a + half]))
    return blocks


def log2_sf_lams(lams, T):
    """log2 of the survival function of sum lam_i * chi^2_1, saddlepoint."""
    lams = [float(x) for x in lams]
    mx = max(lams)

    def Kp(t):
        s = 0.0
        for lam in lams:
            s += lam / (1.0 - 2.0 * lam * t)
        return s

    mean = Kp(0.0)
    if T <= mean:
        # Not a far tail. Crude but only hit if a tilt makes failure likely.
        if T <= 0:
            return 0.0
        # Mills-style fallback is unnecessary for our parameter range.
        return 0.0
    hi = 0.5 / mx * (1.0 - 1e-12)
    lo = 0.0
    if Kp(hi) < T:
        return -1e9
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if Kp(mid) < T:
            lo = mid
        else:
            hi = mid
    t = 0.5 * (lo + hi)
    K = 0.0
    Kpp = 0.0
    for lam in lams:
        d = 1.0 - 2.0 * lam * t
        K += -0.5 * math.log(d)
        Kpp += 2.0 * lam * lam / (d * d)
    log_p = K - t * T - math.log(t) - 0.5 * math.log(2.0 * math.pi * Kpp)
    return log_p / math.log(2.0)


def log2sumexp2(xs):
    m = max(xs)
    if m < -1e8:
        return m
    acc = sum(2.0 ** (x - m) for x in xs)
    return m + math.log2(acc)


class Model:
    def __init__(self):
        self.fac = ring_factors(N)
        self.c2 = all_Q(np.ones(N)) - self.fac
        self.blocks = block_indices(N)
        self.fac_b = np.stack([self.fac[ix] for ix in self.blocks])
        self.c2_b = np.stack([self.c2[ix] for ix in self.blocks])

        self.xs_r, self.p_r = cbd_pmf(ETA_R)
        self.xs_e, self.p_e = cbd_pmf(ETA_E)
        self.xs_g, self.p_g = cbd_pmf(ETA_G)
        self.xs_f, self.p_f = cbd_pmf(ETA_F)
        self.xs_eps, self.p_eps = rounding_pmf(Q, Q2)
        self.codes = e8_codewords()

        self.var_g = moments(self.xs_g, self.p_g)[1]  # mean 0
        self.var_f = moments(self.xs_f, self.p_f)[1]
        self.mu_r0, self.s_r0 = moments(self.xs_r, self.p_r)
        self.mu_e0, self.s_e0 = moments(self.xs_e, self.p_e)
        self.mu_eps, self.s_eps = moments(self.xs_eps, self.p_eps)
        self.var_u = self._var_u(self.p_e)

        # honest code is uniform, gamma = 0
        pr, _ = code_tilt(self.codes, 0.0)
        self.pi0, _, self.bit_bias0 = w_moments_from_code(self.codes, pr)
        self.mu_v0, self.s_v0 = self.v_moments(self.p_e, self.pi0)

    def _var_u(self, p_e):
        # u = e + eps, independent, variance of the sum of squares shift
        mu_e, s_e = moments(self.xs_e, p_e)
        # E[(e+eps)^2] = E[e^2] + E[eps^2] + 2 E[e] E[eps]
        return s_e + self.s_eps + 2 * mu_e * self.mu_eps

    def v_moments(self, p_e, pi):
        """Moments of v = 2(e+eps)+w with w ~ Bern(pi) independent of (e, eps)."""
        mu = 0.0
        sec = 0.0
        for e, pe in zip(self.xs_e, p_e):
            for eps, pp in zip(self.xs_eps, self.p_eps):
                for w, pw in ((0.0, 1.0 - pi), (1.0, pi)):
                    v = 2.0 * (e + eps) + w
                    pr = pe * pp * pw
                    mu += pr * v
                    sec += pr * v * v
        return mu, sec

    def lambdas(self, s_r, mu_r, s_v, mu_v, var_u, pi):
        # r is an i.i.d. product measure, so paired coefficients are uncorrelated.
        # v pairs at distance n/2 share the same Double-E8 bit w, and every
        # cross term of multiplication is exactly such a pair with positive sign.
        # E[Q(v)] = fac * E[v^2] + c2 * (mu_v^2 + pi*(1-pi)).
        Qr = self.fac_b * s_r + self.c2_b * (mu_r ** 2)
        Qv = self.fac_b * s_v + self.c2_b * (mu_v ** 2 + pi * (1.0 - pi))
        return self.var_g * Qr + self.var_f * Qv + var_u

    def log2_delta(self, s_r, mu_r, s_v, mu_v, var_u, pi):
        lams = self.lambdas(s_r, mu_r, s_v, mu_v, var_u, pi)
        logs = [log2_sf_lams(lams[i], T_THRESH) for i in range(N_BLOCKS)]
        return log2sumexp2(logs), logs

    def decompose(self, s_r, mu_r, s_v, mu_v, var_u, pi, block=0):
        Qr = self.fac_b[block] * s_r + self.c2_b[block] * (mu_r ** 2)
        Qv = self.fac_b[block] * s_v + self.c2_b[block] * (
            mu_v ** 2 + pi * (1.0 - pi)
        )
        vr = self.var_g * Qr
        vv = self.var_f * Qv
        return {
            "mean_var_r": float(vr.mean()),
            "mean_var_v": float(vv.mean()),
            "var_u": float(var_u),
            "mean_lambda": float((vr + vv + var_u).mean()),
            "pi": float(pi),
        }


def chi2_log2_sf(k, x):
    """Exact log2 survival of chi^2_k via mpmath, for calibration."""
    from mpmath import mp, mpf, log, gammainc

    mp.dps = 40
    p = gammainc(mpf(k) / 2, mpf(x) / 2, mp.inf, regularized=True)
    return float(log(p) / log(2))


def toy_check(seed=0):
    """Small-ring check: predicted block-failure rate vs direct sampling.

    Same CBD widths as DTRU-2048, n=64, modulus chosen so failures are visible.
    Epsilon is drawn from the compression channel, independent of (e, w), which
    is the model used for the large parameter set (h*r uniform in the ring).
    """
    rng = np.random.default_rng(seed)
    n = 64
    half = n // 2
    q, q2 = 193, 64
    T = (q - 1) ** 2 / 2
    fac = ring_factors(n)
    c2 = all_Q(np.ones(n)) - fac
    blocks = block_indices(n)
    xs_e, p_e = cbd_pmf(ETA_E)
    xs_eps, p_eps = rounding_pmf(q, q2)
    xs_r, p_r = cbd_pmf(ETA_R)
    var_g = cbd_pmf(ETA_G)[0]
    # recompute var properly
    xg, pg = cbd_pmf(ETA_G)
    xf, pf = cbd_pmf(ETA_F)
    var_g = moments(xg, pg)[1]
    var_f = moments(xf, pf)[1]
    mu_e, s_e = moments(xs_e, p_e)
    mu_eps, s_eps = moments(xs_eps, p_eps)
    # w marginal 1/2
    mu_v = 2 * (mu_e + mu_eps) + 0.5
    # E[v^2]
    mu = 0.0
    sec = 0.0
    for e, pe in zip(xs_e, p_e):
        for eps, pp in zip(xs_eps, p_eps):
            for w, pw in ((0.0, 0.5), (1.0, 0.5)):
                v = 2 * (e + eps) + w
                pr = pe * pp * pw
                mu += pr * v
                sec += pr * v * v
    var_u = s_e + s_eps + 2 * mu_e * mu_eps
    pi = 0.5
    # predict, including the shared-w covariance on every cross term
    logs = []
    for ix in blocks:
        Qr = fac[ix] * 1.0  # mu_r = 0, s_r = 1
        Qv = fac[ix] * sec + c2[ix] * (mu ** 2 + pi * (1.0 - pi))
        lams = var_g * Qr + var_f * Qv + var_u
        logs.append(log2_sf_lams(lams, T))
    pred = log2sumexp2(logs)

    def sample_cbd(eta, shape):
        a = rng.integers(0, 2, size=(eta,) + shape)
        b = rng.integers(0, 2, size=(eta,) + shape)
        return (a - b).sum(axis=0).astype(np.float64)

    def mul(a, b):
        acc = np.zeros(n, dtype=np.float64)
        shift = b.copy()
        for j in range(n):
            acc += a[j] * shift
            shift = mul_by_x(shift, half)
        return acc

    trials = 4000
    fails = 0
    sq = np.zeros(n)
    # sample eps by inverse cdf
    cdf = np.cumsum(p_eps)

    def sample_eps(shape):
        u = rng.random(shape)
        idx = np.searchsorted(cdf, u)
        return xs_eps[idx]

    for _ in range(trials):
        g = sample_cbd(ETA_G, (n,))
        fp = sample_cbd(ETA_F, (n,))
        r = sample_cbd(ETA_R, (n,))
        e = sample_cbd(ETA_E, (n,))
        # honest code:  n/16 independent codewords on the first half, then repeat
        w = np.zeros(n, dtype=np.float64)
        codes = e8_codewords()
        for i in range(n // 16):
            c = codes[int(rng.integers(0, 16))]
            w[8 * i : 8 * i + 8] = c
        w[half:] = w[:half]
        eps = sample_eps((n,))
        v = 2 * (e + eps) + w
        u = e + eps
        err = mul(g, r) + mul(fp, v) + u
        sq += err * err
        bad = False
        for ix in blocks:
            if np.dot(err[ix], err[ix]) >= T:
                bad = True
                break
        fails += int(bad)
    rate = fails / trials
    lam = np.zeros(n)
    for k in range(n):
        Qr = fac[k]
        Qv = fac[k] * sec + c2[k] * (mu ** 2 + pi * (1.0 - pi))
        lam[k] = var_g * Qr + var_f * Qv + var_u
    emp = sq / trials
    ratio = emp / lam
    return {
        "n": n,
        "q": q,
        "q2": q2,
        "trials": trials,
        "fails": fails,
        "log2_empirical": math.log2(rate) if rate > 0 else None,
        "log2_predicted": pred,
        "second_moment_ratio_mean": float(ratio.mean()),
        "second_moment_ratio_block0": float(ratio[blocks[0]].mean()),
        "mean_lambda_block0": float(lam[blocks[0]].mean()),
    }


def optimize(model: Model):
    """Grid + local polish of classical work and of quantum work."""

    def pack(theta):
        ar, br, ae, be, gamma = theta
        qr = tilt_pmf(model.xs_r, model.p_r, ar, br)
        qe = tilt_pmf(model.xs_e, model.p_e, ae, be)
        pr, _ = code_tilt(model.codes, gamma)
        mu_r, s_r = moments(model.xs_r, qr)
        mu_e, s_e = moments(model.xs_e, qe)
        pi, _, _ = w_moments_from_code(model.codes, pr)
        mu_v, s_v = model.v_moments(qe, pi)
        var_u = model._var_u(qe)
        log2_d, _ = model.log2_delta(s_r, mu_r, s_v, mu_v, var_u, pi)
        kl = (
            N * kl_bits(qr, model.p_r)
            + N * kl_bits(qe, model.p_e)
            + (N // 16) * kl_bits(pr, np.full(16, 1.0 / 16.0))
        )
        # beta = delta(tilted). alpha = 2^{-kl} up to a sqrt(n) prefactor.
        classical = kl - log2_d
        quantum = 0.5 * kl - log2_d
        queries = -log2_d
        return classical, quantum, queries, kl, log2_d, (mu_r, s_r, mu_v, s_v, pi)

    bounds = [(-1.2, 1.5), (-0.8, 1.4), (-1.5, 2.0), (-1.0, 2.0), (-0.5, 4.0)]

    def run(which):
        def obj(theta):
            c, q, *_ = pack(theta)
            return c if which == "classical" else q

        de = differential_evolution(
            obj,
            bounds,
            seed=1,
            popsize=6,
            mutation=0.6,
            recombination=0.8,
            maxiter=12,
            atol=1e-2,
            tol=1e-2,
            workers=1,
            polish=False,
            updating="immediate",
        )
        loc = minimize(obj, de.x, method="Nelder-Mead")
        theta = loc.x if loc.success else de.x
        vals = pack(theta)
        return {
            "theta": [float(x) for x in theta],
            "classical_bits": vals[0],
            "quantum_bits": vals[1],
            "query_bits": vals[2],
            "kl_bits": vals[3],
            "log2_delta": vals[4],
            "moments": {
                "mu_r": vals[5][0],
                "s_r": vals[5][1],
                "mu_v": vals[5][2],
                "s_v": vals[5][3],
                "pi_w": vals[5][4],
            },
            "de_fun": float(de.fun),
        }

    honest = pack((0.0, 0.0, 0.0, 0.0, 0.0))
    out = {
        "honest": {
            "classical_bits": honest[0],
            "quantum_bits": honest[1],
            "query_bits": honest[2],
            "kl_bits": honest[3],
            "log2_delta": honest[4],
            "moments": {
                "mu_r": honest[5][0],
                "s_r": honest[5][1],
                "mu_v": honest[5][2],
                "s_v": honest[5][3],
                "pi_w": honest[5][4],
            },
        }
    }
    out["best_classical"] = run("classical")
    out["best_quantum"] = run("quantum")
    return out


def monte_carlo_Qv(model: Model, samples=60, seed=2):
    """Honest E[Q(v)] with the real code and repetition, versus the product formula."""
    rng = np.random.default_rng(seed)
    xs_e, p_e = model.xs_e, model.p_e
    cdf_e = np.cumsum(p_e)
    cdf_eps = np.cumsum(model.p_eps)
    acc = np.zeros(N)
    for _ in range(samples):
        ue = rng.random(N)
        ueps = rng.random(N)
        e = model.xs_e[np.searchsorted(cdf_e, ue)]
        eps = model.xs_eps[np.searchsorted(cdf_eps, ueps)]
        w = np.zeros(N)
        for i in range(N // 16):
            c = model.codes[int(rng.integers(0, 16))]
            w[8 * i : 8 * i + 8] = c
        w[HALF:] = w[:HALF]
        v = 2 * (e + eps) + w
        acc += all_Q(v)
    emp = acc / samples
    pi = model.pi0
    formula = model.fac * model.s_v0 + model.c2 * (
        model.mu_v0 ** 2 + pi * (1.0 - pi)
    )
    rel = (emp - formula) / formula
    return {
        "samples": samples,
        "rel_mean": float(rel.mean()),
        "rel_min": float(rel.min()),
        "rel_max": float(rel.max()),
        "rel_block0_mean": float(rel[model.blocks[0]].mean()),
    }


def all_block_grams(poly):
    """Gram[b, i, j] = sum_t (p * x^t)_{block b, i} (p * x^t)_{block b, j}."""
    n = poly.shape[0]
    half = n // 2
    nb = n // 16
    acc = np.array(poly, dtype=np.float64).copy()
    G = np.zeros((nb, 16, 16), dtype=np.float64)
    for _ in range(n):
        lo = acc[:half].reshape(nb, 8)
        hi = acc[half:].reshape(nb, 8)
        vec = np.concatenate([lo, hi], axis=1)
        G += vec[:, :, None] * vec[:, None, :]
        acc = mul_by_x(acc, half)
    return G


def _sym(G):
    return 0.5 * (G + np.swapaxes(G, -1, -2))


def estimate_spectrum_forms(model: Model, samples=32, seed=3):
    """Ring quadratic forms that determine E[Gram] under product measures.

    H is E[Gram(z)] for i.i.d. centered unit-variance coefficients.
    G_ones = Gram(1,1,...,1).
    G_w is E[Gram(w - 1/2)] for an honest Double-E8 codeword polynomial.
    """
    rng = np.random.default_rng(seed)
    H = np.zeros((N_BLOCKS, 16, 16))
    G_w = np.zeros_like(H)
    cdf_e = np.cumsum(model.p_e)
    for _ in range(samples):
        # CBD(2) is exactly mean 0, variance 1
        a = rng.integers(0, 2, size=(ETA_R, N))
        b = rng.integers(0, 2, size=(ETA_R, N))
        r = (a - b).sum(0).astype(np.float64)
        H += all_block_grams(r)
        w = np.zeros(N)
        for i in range(N_BLOCKS):
            w[8 * i : 8 * i + 8] = model.codes[int(rng.integers(0, 16))]
        w[HALF:] = w[:HALF]
        G_w += all_block_grams(w - 0.5)
    H /= samples
    G_w /= samples
    G_ones = all_block_grams(np.ones(N))
    return _sym(H), _sym(G_ones), _sym(G_w)


def spectrum_sigmas(forms, s_r, mu_r, s_a, mu_a, mu_v, pi):
    """Covariance of each 16-dimensional decoder block, in expectation."""
    H, G_ones, G_w = forms
    var_r = s_r - mu_r ** 2
    var_a = s_a - mu_a ** 2
    # honest paired-bit variance is 1/4; scale the centered code Gram with it
    scale_w = (pi * (1.0 - pi)) / 0.25
    Gr = var_r * H + (mu_r ** 2) * G_ones
    Gv = var_a * H + scale_w * G_w + (mu_v ** 2) * G_ones
    return Gr, Gv


def a_moments(model: Model, p_e):
    """Moments of a = 2(e+eps). eps stays at the compression law."""
    mu = 0.0
    sec = 0.0
    for e, pe in zip(model.xs_e, p_e):
        for eps, pp in zip(model.xs_eps, model.p_eps):
            a = 2.0 * (e + eps)
            pr = float(pe * pp)
            mu += pr * a
            sec += pr * a * a
    return mu, sec


def log2_delta_from_sigmas(model, Gr, Gv, var_u):
    logs = []
    eye = np.eye(16)
    for b in range(N_BLOCKS):
        Sig = model.var_g * Gr[b] + model.var_f * Gv[b] + var_u * eye
        ev = np.linalg.eigvalsh(Sig)
        ev = np.clip(ev, 1e-9, None)
        logs.append(log2_sf_lams(ev, T_THRESH))
    return log2sumexp2(logs), logs


def sample_average_delta(model: Model, samples=16, seed=4):
    """Monte Carlo estimate of E[delta], averaging the tail over fresh (r, v)."""
    rng = np.random.default_rng(seed)
    cdf_e = np.cumsum(model.p_e)
    cdf_eps = np.cumsum(model.p_eps)
    # per-block list of probabilities
    acc = np.zeros(N_BLOCKS)
    eye = np.eye(16)
    for _ in range(samples):
        a = rng.integers(0, 2, size=(ETA_R, N))
        b = rng.integers(0, 2, size=(ETA_R, N))
        r = (a - b).sum(0).astype(np.float64)
        e = model.xs_e[np.searchsorted(cdf_e, rng.random(N))]
        eps = model.xs_eps[np.searchsorted(cdf_eps, rng.random(N))]
        w = np.zeros(N)
        for i in range(N_BLOCKS):
            w[8 * i : 8 * i + 8] = model.codes[int(rng.integers(0, 16))]
        w[HALF:] = w[:HALF]
        v = 2.0 * (e + eps) + w
        Gr = all_block_grams(r)
        Gv = all_block_grams(v)
        for b in range(N_BLOCKS):
            Sig = model.var_g * Gr[b] + model.var_f * Gv[b] + model.var_u * eye
            ev = np.clip(np.linalg.eigvalsh(Sig), 1e-9, None)
            acc[b] += 2.0 ** log2_sf_lams(ev, T_THRESH)
    acc /= samples
    # numerical floor
    acc = np.maximum(acc, 2.0 ** -1000)
    logs = np.log2(acc)
    return float(log2sumexp2(logs.tolist())), logs.tolist()


def optimize_spectrum(model: Model, forms):
    """Failure boosting where the score is the spectrum of the block covariance."""

    def pack(theta):
        ar, br, ae, be, gamma = [float(x) for x in theta]
        qr = tilt_pmf(model.xs_r, model.p_r, ar, br)
        qe = tilt_pmf(model.xs_e, model.p_e, ae, be)
        pr, _ = code_tilt(model.codes, gamma)
        mu_r, s_r = moments(model.xs_r, qr)
        mu_a, s_a = a_moments(model, qe)
        pi, _, _ = w_moments_from_code(model.codes, pr)
        mu_v = mu_a + pi
        var_u = model._var_u(qe)
        Gr, Gv = spectrum_sigmas(forms, s_r, mu_r, s_a, mu_a, mu_v, pi)
        log2_d, logs = log2_delta_from_sigmas(model, Gr, Gv, var_u)
        kl = (
            N * kl_bits(qr, model.p_r)
            + N * kl_bits(qe, model.p_e)
            + (N // 16) * kl_bits(pr, np.full(16, 1.0 / 16.0))
        )
        return {
            "classical_bits": kl - log2_d,
            "quantum_bits": 0.5 * kl - log2_d,
            "query_bits": -log2_d,
            "kl_bits": kl,
            "log2_delta": log2_d,
            "block0_log2": logs[0],
            "mu_r": mu_r,
            "s_r": s_r,
            "mu_v": mu_v,
            "s_a": s_a,
            "pi_w": pi,
            "theta": [ar, br, ae, be, gamma],
        }

    bounds = [(-0.8, 1.2), (-0.4, 1.2), (-0.8, 1.6), (-0.5, 1.6), (-0.2, 2.5)]

    def run(which):
        def obj(theta):
            info = pack(theta)
            return info["classical_bits"] if which == "classical" else info["quantum_bits"]

        de = differential_evolution(
            obj,
            bounds,
            seed=2,
            popsize=6,
            mutation=0.5,
            recombination=0.7,
            maxiter=10,
            atol=1e-2,
            tol=1e-2,
            workers=1,
            polish=False,
            updating="immediate",
        )
        loc = minimize(obj, de.x, method="Nelder-Mead", options={"maxiter": 80})
        theta = loc.x if np.isfinite(loc.fun) else de.x
        info = pack(theta)
        info["de_fun"] = float(de.fun)
        return info

    honest = pack((0.0, 0.0, 0.0, 0.0, 0.0))
    return {
        "honest": honest,
        "best_classical": run("classical"),
        "best_quantum": run("quantum"),
    }


def implied_sigma_for_claim():
    """sigma^2 that makes (n/16) * P(chi^2_16 >= T/sigma^2) = 2^{-204.10}."""
    target = -204.10 - math.log2(N_BLOCKS)  # per-block
    lo, hi = 200.0, 800.0
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        val = chi2_log2_sf(16, mid)
        if val > target:
            lo = mid
        else:
            hi = mid
    x = 0.5 * (lo + hi)
    return {"chi2_threshold": x, "sigma2": T_THRESH / x, "per_block_log2": target}


def main():
    model = Model()
    claim = implied_sigma_for_claim()

    # calibration: 16 equal variances, saddlepoint vs exact chi^2
    sig2 = claim["sigma2"]
    saddle_one = log2_sf_lams([sig2] * 16, T_THRESH)
    exact_one = chi2_log2_sf(16, T_THRESH / sig2)

    log2_d, logs = model.log2_delta(
        model.s_r0, model.mu_r0, model.s_v0, model.mu_v0, model.var_u, model.pi0
    )
    # centered-v variant: drop (E v)^2 and the shared-bit covariance
    log2_centered, logs_c = model.log2_delta(
        model.s_r0, 0.0, model.s_v0, 0.0, model.var_u, 0.0
    )
    # designer-style: every coefficient uses the upper-half factor 3n/2,
    # and the cross term is ignored
    fac_hi = 1.5 * N
    # match their union bound with a single sigma^2 built from our moments
    # sigma^2 = var_g * fac_hi * s_r + var_f * fac_hi * s_v + var_u, mu=0
    sig_hi = model.var_g * fac_hi * model.s_r0 + model.var_f * fac_hi * model.s_v0 + model.var_u
    log_hi = chi2_log2_sf(16, T_THRESH / sig_hi) + math.log2(N_BLOCKS)

    dec0 = model.decompose(
        model.s_r0, model.mu_r0, model.s_v0, model.mu_v0, model.var_u, model.pi0, 0
    )
    dec_mid = model.decompose(
        model.s_r0, model.mu_r0, model.s_v0, model.mu_v0, model.var_u, model.pi0, 64
    )

    print("rounding pmf", list(zip(model.xs_eps.tolist(), model.p_eps.tolist())))
    print("E[eps], E[eps^2]", model.mu_eps, model.s_eps)
    print("honest v moments", model.mu_v0, model.s_v0, "var_u", model.var_u)
    print("var g,f,r,e", model.var_g, model.var_f, model.s_r0, model.s_e0)
    print("c2 min/mean/max", float(model.c2.min()), float(model.c2.mean()), float(model.c2.max()))
    print("block log2 tail first/mid/last", logs[0], logs[64], logs[-1])
    print("log2 delta position-dependent", log2_d)
    print("log2 delta mu forced 0", log2_centered)
    print("log2 delta fac=3n/2 chi2 union", log_hi, "sigma2", sig_hi)
    print("claim sigma2", claim)
    print("saddle vs exact at claim sigma", saddle_one, exact_one)
    print("decompose block0", dec0)
    print("decompose block64", dec_mid)

    print("monte carlo Q(v) ...")
    mc = monte_carlo_Qv(model)
    print("mc", mc)

    print("toy check ...")
    toy = toy_check()
    print("toy", toy)

    print("independence-model optimizing ...")
    opt = optimize(model)
    print(json.dumps(opt, indent=2))

    print("estimating block-covariance forms ...")
    forms = estimate_spectrum_forms(model, samples=24, seed=3)
    H, G_ones, G_w = forms
    # trace check: for unit-variance centered r, trace of block 0 equals sum of factors
    fac_b0 = float(model.fac[model.blocks[0]].sum())
    print("trace H block0", float(np.trace(H[0])), "sum fac", fac_b0)

    print("spectrum optimizing ...")
    spec = optimize_spectrum(model, forms)
    # drop nothing; print compact
    def brief(info):
        return {
            k: info[k]
            for k in (
                "classical_bits",
                "quantum_bits",
                "query_bits",
                "kl_bits",
                "log2_delta",
                "block0_log2",
                "mu_r",
                "s_r",
                "mu_v",
                "s_a",
                "pi_w",
                "theta",
            )
            if k in info
        }

    print("spectrum honest", json.dumps(brief(spec["honest"]), indent=2))
    print("spectrum classical", json.dumps(brief(spec["best_classical"]), indent=2))
    print("spectrum quantum", json.dumps(brief(spec["best_quantum"]), indent=2))

    print("monte carlo E[delta] ...")
    mc_delta, mc_logs = sample_average_delta(model, samples=8, seed=5)
    print("E[delta] log2", mc_delta, "worst sample-avg block", max(mc_logs))

    pref = 0.5 * math.log2(2 * math.pi * N)
    result = {
        "parameters": {
            "n": N,
            "q": Q,
            "q2": Q2,
            "threshold_T": T_THRESH,
            "eta": {"g": ETA_G, "f": ETA_F, "r": ETA_R, "e": ETA_E},
            "var_g": model.var_g,
            "var_f": model.var_f,
            "var_r": model.s_r0,
            "var_e": model.s_e0,
            "mu_eps": model.mu_eps,
            "second_eps": model.s_eps,
            "mu_v": model.mu_v0,
            "second_v": model.s_v0,
            "var_u": model.var_u,
        },
        "baseline": {
            "log2_delta_position_dependent": log2_d,
            "log2_delta_mean_zero": log2_centered,
            "log2_delta_fac_3n_over_2": log_hi,
            "sigma2_fac_3n_over_2": sig_hi,
            "claimed": -204.10,
            "implied_sigma2": claim,
            "saddle_minus_exact_bits": saddle_one - exact_one,
            "block0": dec0,
            "block0_log2": logs[0],
            "block64_log2": logs[64],
            "block127_log2": logs[-1],
            "epsilon_pmf": {
                str(int(x)): float(p) for x, p in zip(model.xs_eps, model.p_eps)
            },
        },
        "correlation_check": mc,
        "toy": toy,
        "boost_independence_model": opt,
        "spectrum": {
            "honest": brief(spec["honest"]),
            "best_classical": brief(spec["best_classical"]),
            "best_quantum": brief(spec["best_quantum"]),
            "monte_carlo_log2_E_delta": mc_delta,
            "trace_H_block0": float(np.trace(H[0])),
            "sum_fac_block0": fac_b0,
            "eig_block0_honest": np.linalg.eigvalsh(
                model.var_g * spectrum_sigmas(
                    forms,
                    model.s_r0,
                    model.mu_r0,
                    a_moments(model, model.p_e)[1],
                    a_moments(model, model.p_e)[0],
                    model.mu_v0,
                    model.pi0,
                )[0][0]
                + model.var_f * spectrum_sigmas(
                    forms,
                    model.s_r0,
                    model.mu_r0,
                    a_moments(model, model.p_e)[1],
                    a_moments(model, model.p_e)[0],
                    model.mu_v0,
                    model.pi0,
                )[1][0]
                + model.var_u * np.eye(16)
            ).tolist(),
        },
        "prefactor_half_log2_2pin_bits": pref,
        "security_claim": {"classical_core_svp": 567, "quantum_core_svp": 498, "nominal": 512},
    }
    out = Path(__file__).resolve().parent / "failure_boost_results.json"
    out.write_text(json.dumps(result, indent=2))
    print("wrote", out)


def tail_survey(samples=40, seed=7):
    """Distribution of per-ciphertext failure probabilities under the spectrum model."""
    rng = np.random.default_rng(seed)
    model = Model()
    cdf_e = np.cumsum(model.p_e)
    cdf_eps = np.cumsum(model.p_eps)
    eye = np.eye(16)
    rows = []
    for _ in range(samples):
        a = rng.integers(0, 2, size=(ETA_R, N))
        b = rng.integers(0, 2, size=(ETA_R, N))
        r = (a - b).sum(0).astype(np.float64)
        e = model.xs_e[np.searchsorted(cdf_e, rng.random(N))]
        eps = model.xs_eps[np.searchsorted(cdf_eps, rng.random(N))]
        w = np.zeros(N)
        for i in range(N_BLOCKS):
            w[8 * i : 8 * i + 8] = model.codes[int(rng.integers(0, 16))]
        w[HALF:] = w[:HALF]
        v = 2.0 * (e + eps) + w
        Gr = all_block_grams(r)
        Gv = all_block_grams(v)
        logs = []
        top = 0.0
        for b in range(N_BLOCKS):
            Sig = model.var_g * Gr[b] + model.var_f * Gv[b] + model.var_u * eye
            ev = np.linalg.eigvalsh(Sig)
            top = max(top, float(ev[-1]))
            logs.append(log2_sf_lams(np.clip(ev, 1e-9, None), T_THRESH))
        rows.append(
            {
                "log2_p": log2sumexp2(logs),
                "top_eigenvalue": top,
                "mean_square_r": float(np.dot(r, r) / N),
                "block0_log2": logs[0],
            }
        )
    lp = np.array([row["log2_p"] for row in rows])
    top = np.array([row["top_eigenvalue"] for row in rows])
    summary = {
        "samples": samples,
        "mean_log2": float(lp.mean()),
        "std_log2": float(lp.std(ddof=1)),
        "median_log2": float(np.median(lp)),
        "min_log2": float(lp.min()),
        "max_log2": float(lp.max()),
        "log2_mean_p": float(np.log2(np.exp2(lp).mean())),
        "corr_log2_vs_top_eigenvalue": float(np.corrcoef(lp, top)[0, 1]),
        "gaussian_log2_E_p": float(
            lp.mean() + (lp.std(ddof=1) ** 2) * math.log(2) / 2
        ),
    }
    out = Path(__file__).resolve().parent / "failure_tail.json"
    out.write_text(json.dumps({"summary": summary, "samples": rows}, indent=2))
    print(json.dumps(summary, indent=2))
    print("wrote", out)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "tail":
        tail_survey()
    else:
        main()
