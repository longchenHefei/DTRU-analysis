"""What a DTRU decryption failure reveals, and a spectral key-recovery test.

Failure event (Algorithm 2, symbols 0 and (q+1)/2 are antipodal in Z_q):
    exists codeword-difference support D:  sum_{j in D} |Err_j| >= |D| (q+1)/4,
i.e. the union over ALL sign patterns s on D of  s . Err >= |D|(q+1)/4.
(A fixed-sign version misses almost every failure; see `check`.)

Given the ciphertext, s . Err = a_g . g + a_f . f' + s . u with a_g, a_f known.
A failure therefore says |a . x| is large for the most likely pattern s; the
sign of a . x is unknown. The secret x = (g, f') can be estimated up to a
global sign by the top eigenvector of  sum_k a_k a_k^T  over failures k.

    python3 attack/DTRU/recover.py check   # predicate vs Algorithm 2
    python3 attack/DTRU/recover.py toy     # n=64: real failures -> spectral recovery
    python3 attack/DTRU/recover.py full    # n=2048: alignment between failure directions
"""

import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy.special import log_ndtr

import failure_boost as fb
from ml_decoder_check import DIRS, SIZES, decode_block


def embed(u8):
    return np.concatenate([u8, u8]).astype(np.float64)


def mul_matrix(poly):
    n = poly.shape[0]
    half = n // 2
    M = np.empty((n, n), dtype=np.float64)
    acc = np.array(poly, dtype=np.float64).copy()
    for j in range(n):
        M[:, j] = acc
        acc = fb.mul_by_x(acc, half)
    return M


def sample_cbd(rng, eta, shape):
    a = rng.integers(0, 2, size=(eta,) + shape)
    b = rng.integers(0, 2, size=(eta,) + shape)
    return (a - b).sum(0).astype(np.float64)


def block_idx(n, b):
    half = n // 2
    return np.concatenate([np.arange(8 * b, 8 * b + 8), np.arange(8 * b + half, 8 * b + 8 + half)])


def check(trials=2000, seed=5):
    rng = np.random.default_rng(seed)
    codes = fb.e8_codewords()
    out = {}
    for q, scales in ((193, (30, 40)), (3457, (700, 1000))):
        h = (q + 1) // 2
        for sc in scales:
            agree = pred_only = dec_only = fails = 0
            for _ in range(trials):
                err = rng.normal(0, sc, size=16)
                if np.max(np.abs(err)) > q / 2 - 3:
                    continue
                sent = int(rng.integers(0, 16))
                rec = np.mod(h * embed(codes[sent]) + np.rint(err), q)
                dec = decode_block(rec, q) != sent
                hs = bool(np.any(np.abs(DIRS @ err) >= SIZES * (q + 1) / 4))
                fails += dec
                if dec == hs:
                    agree += 1
                elif hs:
                    pred_only += 1
                else:
                    dec_only += 1
            out[f"q{q}_scale{sc}"] = {
                "decoder_fails": fails, "agree": agree,
                "predicate_only": pred_only, "decoder_only": dec_only,
            }
    return out


def toy(target_fails=150, batch=4000, seed=3):
    """n=64, q=193: real Algorithm 2 failures on one key, then spectral recovery."""
    rng = np.random.default_rng(seed)
    n, half, q, q2 = 64, 32, 193, 64
    nb = n // 16
    xs_e, p_e = fb.cbd_pmf(fb.ETA_E)
    xs_eps, p_eps = fb.rounding_pmf(q, q2)
    codes = fb.e8_codewords()
    g = sample_cbd(rng, fb.ETA_G, (n,))
    fp = sample_cbd(rng, fb.ETA_F, (n,))
    x_true = np.concatenate([g, fp])
    Mg, Mf = mul_matrix(g), mul_matrix(fp)
    var_g, var_f = fb.ETA_G / 2, fb.ETA_F / 2
    thr = SIZES * (q + 1) / 4.0
    h = (q + 1) // 2

    seen = 0
    fails = 0
    A = []       # observation directions a (2n)
    zs = []      # attacker-side z of the chosen pattern
    right_pattern = 0
    while fails < target_fails:
        r = sample_cbd(rng, fb.ETA_R, (n, batch))
        e = xs_e[np.searchsorted(np.cumsum(p_e), rng.random((n, batch)))]
        eps = xs_eps[np.searchsorted(np.cumsum(p_eps), rng.random((n, batch)))]
        msg = rng.integers(0, 16, size=(nb, batch))
        w = np.zeros((n, batch))
        for b in range(nb):
            cw = np.stack([codes[m] for m in msg[b]], axis=1)
            w[8 * b : 8 * b + 8] = cw
        w[half:] = w[:half]
        v = 2 * (e + eps) + w
        u = e + eps
        err = Mg @ r + Mf @ v + u
        seen += batch
        for b in range(nb):
            idx = block_idx(n, b)
            eb = err[idx]                          # 16 x batch
            score = np.abs(DIRS @ eb) - thr[:, None]
            bad = np.flatnonzero(score.max(axis=0) >= 0)
            for t in bad:
                # confirm with the real decoder
                rec = np.mod(h * embed(codes[int(msg[b, t])]) + np.rint(eb[:, t]), q)
                if decode_block(rec, q) == int(msg[b, t]):
                    continue
                fails += 1
                # attacker: covariance of this block given (r, v), pick most likely pattern
                Mr = mul_matrix(r[:, t])[idx]      # 16 x n
                Mv = mul_matrix(v[:, t])[idx]
                Sig = var_g * Mr @ Mr.T + var_f * Mv @ Mv.T
                sd = np.sqrt(np.einsum("ki,ij,kj->k", DIRS, Sig, DIRS))
                z = (thr - DIRS @ u[idx, t]) / sd  # sign-symmetric approx
                k = int(np.argmin(z))
                true_k = int(np.argmax(score[:, t]))
                right_pattern += int(k == true_k)
                s = DIRS[k]
                a = np.concatenate([Mr.T @ s, Mv.T @ s])
                A.append(a / np.linalg.norm(a))
                zs.append(float(z[k]))
                if fails >= target_fails:
                    break
            if fails >= target_fails:
                break

    A = np.array(A)
    # spectral estimate: top eigenvector of sum a a^T (sign-free)
    C = A.T @ A
    evals, evecs = np.linalg.eigh(C)
    xhat = evecs[:, -1]
    corr = abs(float(xhat @ x_true) / np.linalg.norm(x_true))
    # per-observation alignment
    cos_true = np.abs(A @ x_true) / np.linalg.norm(x_true)
    # random-direction baseline
    R = rng.normal(size=(A.shape[0], 2 * n))
    R /= np.linalg.norm(R, axis=1, keepdims=True)
    base = np.abs(R @ x_true) / np.linalg.norm(x_true)
    return {
        "n": n, "q": q, "encapsulations": seen, "decoder_failures": fails,
        "log2_failure_rate": math.log2(fails / seen),
        "chosen_pattern_matches_realized": right_pattern / fails,
        "mean_abs_cos_direction_secret": float(cos_true.mean()),
        "random_direction_baseline": float(base.mean()),
        "spectral_estimate_corr_with_secret": corr,
        "attacker_z_median": float(np.median(zs)),
    }


def full(samples=24, seed=7):
    """n=2048: per ciphertext, the most likely failure pattern and its direction."""
    rng = np.random.default_rng(seed)
    n, half, q = fb.N, fb.N // 2, fb.Q
    nb = n // 16
    var_g, var_f = fb.ETA_G / 2, fb.ETA_F / 2
    xs_e, p_e = fb.cbd_pmf(fb.ETA_E)
    xs_eps, p_eps = fb.rounding_pmf(q, fb.Q2)
    codes = fb.e8_codewords()
    thr = SIZES * (q + 1) / 4.0
    dirs = []
    rows = []
    for _ in range(samples):
        r = sample_cbd(rng, fb.ETA_R, (n,))
        e = xs_e[np.searchsorted(np.cumsum(p_e), rng.random(n))]
        eps = xs_eps[np.searchsorted(np.cumsum(p_eps), rng.random(n))]
        msg = rng.integers(0, 16, size=nb)
        w = np.zeros(n)
        for b, m in enumerate(msg):
            w[8 * b : 8 * b + 8] = codes[int(m)]
        w[half:] = w[:half]
        v = 2 * (e + eps) + w
        u = e + eps
        Mr, Mv = mul_matrix(r), mul_matrix(v)
        best = None
        logs = []
        for b in range(nb):
            idx = block_idx(n, b)
            Sig = var_g * Mr[idx] @ Mr[idx].T + var_f * Mv[idx] @ Mv[idx].T
            sd = np.sqrt(np.einsum("ki,ij,kj->k", DIRS, Sig, DIRS))
            lg = np.logaddexp(log_ndtr(-(thr - DIRS @ u[idx]) / sd),
                              log_ndtr(-(thr + DIRS @ u[idx]) / sd)) / math.log(2)
            logs.append(fb.log2sumexp2(lg.tolist()))
            k = int(np.argmax(lg))
            if best is None or lg[k] > best[0]:
                s = DIRS[k]
                a = np.concatenate([Mr[idx].T @ s, Mv[idx].T @ s])
                best = (float(lg[k]), a, float(thr[k] / sd[k]))
        a = best[1]
        en = a * a
        neff = float(en.sum() ** 2 / np.sum(en * en))
        dirs.append(a / np.linalg.norm(a))
        p = fb.log2sumexp2(logs)
        rows.append({"log2_p": p, "top_pattern_log2": best[0], "z": best[2], "neff": neff})
        print(f"{p:8.2f} top {best[0]:8.2f} z {best[2]:5.2f} neff {neff:7.1f}", flush=True)
    D = np.array(dirs)
    G = np.abs(D @ D.T)
    off = G[np.triu_indices(samples, 1)]
    lp = np.array([r["log2_p"] for r in rows])
    summary = {
        "samples": samples,
        "median_log2_p": float(np.median(lp)),
        "log2_mean_p": float(np.log2(np.mean(np.exp2(lp)))),
        "z_median": float(np.median([r["z"] for r in rows])),
        "neff_median": float(np.median([r["neff"] for r in rows])),
        "abs_cosine_between_failure_directions_median": float(np.median(off)),
        "abs_cosine_max": float(off.max()),
    }
    out = Path(__file__).resolve().parent / "recover_full.json"
    out.write_text(json.dumps({"summary": summary, "samples": rows}, indent=2))
    return summary


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "check"
    fn = {"check": check, "toy": toy, "full": full}[cmd]
    print(json.dumps(fn(), indent=2))
