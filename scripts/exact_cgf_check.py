"""ML-decoder failure with the exact CBD distribution of the secret.

ml_decoder_check.py treats Err | (r, v) as Gaussian. Here each half-space
s^T Err >= |D|(q+1)/4 is evaluated with the exact cumulant generating function
of the linear form sum_j g_j A_j + f'_j B_j, g ~ CBD(5), f' ~ CBD(1),
K(t) = sum_j 2 eta log cosh(t a_j / 2), via Lugannani-Rice. Only the TOP
Gaussian terms per ciphertext are recomputed; the rest keep the Gaussian value.

    python3 attack/DTRU/exact_cgf_check.py toy
    python3 attack/DTRU/exact_cgf_check.py full
"""

import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import brentq
from scipy.special import log_ndtr

import failure_boost as fb
from ml_decoder_check import DIRS, SIZES

TOP = 64


def block_maps(poly):
    """M[b, k, j] = (poly * x^j)_{block b, local k}."""
    n = poly.shape[0]
    half, nb = n // 2, n // 16
    acc = np.array(poly, dtype=np.float64).copy()
    M = np.empty((nb, 16, n))
    for j in range(n):
        M[:, :8, j] = acc[:half].reshape(nb, 8)
        M[:, 8:, j] = acc[half:].reshape(nb, 8)
        acc = fb.mul_by_x(acc, half)
    return M


def log_lr_tail(a, eta, c):
    """log P(sum_j X_j a_j >= c), X_j ~ CBD(eta_j) independent (Lugannani-Rice)."""
    def K(t):
        x = np.abs(t * a / 2.0)
        return float(np.sum(2 * eta * (x + np.log1p(np.exp(-2 * x)) - math.log(2.0))))

    def K1(t):
        return float(np.sum(eta * a * np.tanh(t * a / 2.0)))

    def K2(t):
        return float(np.sum(eta * a * a / 2.0 / np.cosh(t * a / 2.0) ** 2))

    if c >= float(np.sum(eta * np.abs(a))) * 2 - 1e-9:
        return -math.inf
    hi = 1e-6
    while K1(hi) < c:
        hi *= 2
    th = brentq(lambda t: K1(t) - c, 0.0, hi)
    w = math.sqrt(max(2 * (th * c - K(th)), 1e-300))
    u = th * math.sqrt(K2(th))
    logQ = float(log_ndtr(-w))
    logphi = -w * w / 2 - 0.5 * math.log(2 * math.pi)
    corr = 1.0 + math.exp(logphi - logQ) * (1.0 / u - 1.0 / w)
    return logQ + math.log(max(corr, 1e-300))


def ciphertext_log2(r, v, q, var_g, var_f, top=TOP):
    Mr, Mv = block_maps(r), block_maps(v)
    nb = Mr.shape[0]
    eta = np.concatenate([np.full(Mr.shape[2], fb.ETA_G), np.full(Mv.shape[2], fb.ETA_F)])
    c = SIZES * (q + 1) / 4.0
    gauss = []  # (log natural, block, dir)
    for b in range(nb):
        Sig = var_g * Mr[b] @ Mr[b].T + var_f * Mv[b] @ Mv[b].T
        var = np.einsum("ki,ij,kj->k", DIRS, Sig, DIRS)
        lg = log_ndtr(-c / np.sqrt(var)) + math.log(2.0)
        for k in np.argsort(lg)[-top:]:
            gauss.append((float(lg[k]), b, int(k)))
        rest = np.sort(lg)[:-top]
        gauss.append((float(np.logaddexp.reduce(rest)) if rest.size else -math.inf, b, -1))
    gauss.sort(reverse=True)
    exact, gsum = [], []
    done = 0
    for lg, b, k in gauss:
        gsum.append(lg)
        if k < 0 or done >= top:
            exact.append(lg)
            continue
        done += 1
        s = DIRS[k]
        a = np.concatenate([s @ Mr[b], s @ Mv[b]])
        exact.append(log_lr_tail(a, eta, c[k]) + math.log(2.0))
    to2 = 1 / math.log(2.0)
    return float(np.logaddexp.reduce(gsum)) * to2, float(np.logaddexp.reduce(exact)) * to2


def sample_honest(rng, n, q, q2, codes):
    half = n // 2
    xs_e, p_e = fb.cbd_pmf(fb.ETA_E)
    xs_eps, p_eps = fb.rounding_pmf(q, q2)
    a = rng.integers(0, 2, size=(fb.ETA_R, n))
    b = rng.integers(0, 2, size=(fb.ETA_R, n))
    r = (a - b).sum(0).astype(np.float64)
    e = xs_e[np.searchsorted(np.cumsum(p_e), rng.random(n))]
    eps = xs_eps[np.searchsorted(np.cumsum(p_eps), rng.random(n))]
    w = np.zeros(n)
    for i in range(n // 16):
        w[8 * i : 8 * i + 8] = codes[int(rng.integers(0, 16))]
    w[half:] = w[:half]
    return r, 2.0 * (e + eps) + w


def run(n, q, q2, samples, seed, top):
    rng = np.random.default_rng(seed)
    var_g = fb.moments(*fb.cbd_pmf(fb.ETA_G))[1]
    var_f = fb.moments(*fb.cbd_pmf(fb.ETA_F))[1]
    codes = fb.e8_codewords()
    rows = []
    for _ in range(samples):
        r, v = sample_honest(rng, n, q, q2, codes)
        g, x = ciphertext_log2(r, v, q, var_g, var_f, top)
        rows.append({"gauss": g, "exact": x})
        if n > 64:
            print(f"{g:8.2f} {x:8.2f}", flush=True)
    G = np.array([x["gauss"] for x in rows])
    X = np.array([x["exact"] for x in rows])
    return {
        "n": n, "q": q, "samples": samples,
        "gauss_log2_mean_p": float(np.log2(np.mean(np.exp2(G)))),
        "exact_log2_mean_p": float(np.log2(np.mean(np.exp2(X)))),
        "gauss_median": float(np.median(G)),
        "exact_median": float(np.median(X)),
        "exact_mean_log2": float(X.mean()),
        "exact_std_log2": float(X.std(ddof=1)),
        "exact_min": float(X.min()),
        "exact_max": float(X.max()),
        "median_shift_bits": float(np.median(X - G)),
    }, rows


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "full":
        s, rows = run(fb.N, fb.Q, fb.Q2, 40, 7, TOP)
        out = Path(__file__).resolve().parent / "exact_cgf_results.json"
        out.write_text(json.dumps({"summary": s, "samples": rows}, indent=2))
    else:
        s, _ = run(64, 193, 64, 2000, 11, 300)
    print(json.dumps(s, indent=2))
