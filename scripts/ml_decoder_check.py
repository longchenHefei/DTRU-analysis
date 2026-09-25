"""Failure probability of DTRU with the actual ML decoder (Algorithm 2).

The ball ||Err_block||^2 >= (q-1)^2/2 used in failure_boost.py is only the
sufficient condition of the specification. Algorithm 2 picks the nearest of 16
codewords (q+1)/2 (u, u); it errs only if some wrong codeword is closer. For a
codeword difference supported on D (|D| = 8 or 16) this is

    sum_{j in D} |e_j| >= |D| (q+1)/4,

i.e. a union of half-spaces s^T e >= |D|(q+1)/4 over the 2^|D| sign vectors s
on D. Given (r, v), Err is Gaussian with covariance Sigma = var_g Gram(r) +
var_f Gram(v), so each half-space has probability Q(c / sqrt(s^T Sigma s)).

    python3 attack/DTRU/ml_decoder_check.py toy     # real decoder vs models, n=64
    python3 attack/DTRU/ml_decoder_check.py full    # n=2048, 40 ciphertexts
"""

import itertools
import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy.special import log_ndtr

import failure_boost as fb


def diff_supports():
    """Supports D (block-local 0..15) of nonzero codeword differences (d, d)."""
    sups = []
    for c in fb.e8_codewords():
        idx = np.flatnonzero(c)
        if idx.size:
            sups.append(np.concatenate([idx, idx + 8]))
    return sups


def sign_directions():
    """All (s, c_scale) with s in {0,+-1}^16 supported on some D; one per +-pair."""
    dirs = []
    sizes = []
    for D in diff_supports():
        for signs in itertools.product((1.0, -1.0), repeat=D.size - 1):
            s = np.zeros(16)
            s[D] = (1.0,) + signs
            dirs.append(s)
            sizes.append(D.size)
    return np.array(dirs), np.array(sizes, dtype=np.float64)


DIRS, SIZES = sign_directions()


def log2_ml_block(Sig, q):
    """log2 of the union bound over half-spaces, both signs of each s."""
    var = np.einsum("ki,ij,kj->k", DIRS, Sig, DIRS)
    c = SIZES * (q + 1) / 4.0
    z = c / np.sqrt(var)
    logs = (log_ndtr(-z) + math.log(2.0)) / math.log(2.0)
    return fb.log2sumexp2(logs.tolist())


def decode_block(v, q):
    """Algorithm 2 on one 16-vector v in Z_q; returns index of chosen codeword."""
    half = (q + 1) // 2

    def d2(x):
        x = np.mod(x, q)
        x = np.minimum(x, q - x)
        return x * x

    dis0 = d2(v[:8]) + d2(v[8:])
    dis1 = d2(v[:8] - half) + d2(v[8:] - half)
    best, arg = None, 0
    for j, u in enumerate(fb.e8_codewords()):
        cost = np.where(u == 1, dis1, dis0).sum()
        if best is None or cost < best:
            best, arg = cost, j
    return arg


def toy(q=193, q2=64, trials=20000, seed=11):
    rng = np.random.default_rng(seed)
    n, half = 64, 32
    blocks = fb.block_indices(n)
    xg, pg = fb.cbd_pmf(fb.ETA_G)
    xf, pf = fb.cbd_pmf(fb.ETA_F)
    var_g = fb.moments(xg, pg)[1]
    var_f = fb.moments(xf, pf)[1]
    xs_eps, p_eps = fb.rounding_pmf(q, q2)
    cdf = np.cumsum(p_eps)
    codes = fb.e8_codewords()
    T = (q - 1) ** 2 / 2

    def cbd(eta):
        a = rng.integers(0, 2, size=(eta, n))
        b = rng.integers(0, 2, size=(eta, n))
        return (a - b).sum(0).astype(np.float64)

    def mul(a, b):
        acc = np.zeros(n)
        sh = b.copy()
        for j in range(n):
            acc += a[j] * sh
            sh = fb.mul_by_x(sh, half)
        return acc

    ball_fail = ml_fail = 0
    pred_ball, pred_ml = [], []
    for _ in range(trials):
        g, fp, r, e = cbd(fb.ETA_G), cbd(fb.ETA_F), cbd(fb.ETA_R), cbd(fb.ETA_E)
        msg = rng.integers(0, 16, size=n // 16)
        w = np.zeros(n)
        for i, m in enumerate(msg):
            w[8 * i : 8 * i + 8] = codes[m]
        w[half:] = w[:half]
        eps = xs_eps[np.searchsorted(cdf, rng.random(n))]
        v = 2 * (e + eps) + w
        u = e + eps
        err = mul(g, r) + mul(fp, v) + u
        t = np.mod((q + 1) // 2 * w + np.rint(err), q)
        bf = mf = False
        for i, ix in enumerate(blocks):
            if np.dot(err[ix], err[ix]) >= T:
                bf = True
            if decode_block(t[ix], q) != msg[i]:
                mf = True
        ball_fail += bf
        ml_fail += mf
        Gr = fb.all_block_grams(r)
        Gv = fb.all_block_grams(v)
        lb, lm = [], []
        for b in range(len(blocks)):
            Sig = var_g * Gr[b] + var_f * Gv[b]
            ev = np.clip(np.linalg.eigvalsh(Sig), 1e-9, None)
            lb.append(fb.log2_sf_lams(ev, T))
            lm.append(log2_ml_block(Sig, q))
        pred_ball.append(fb.log2sumexp2(lb))
        pred_ml.append(fb.log2sumexp2(lm))

    def mean_log2(xs):
        return float(np.log2(np.mean(np.exp2(np.minimum(xs, 0.0)))))

    return {
        "n": n, "q": q, "q2": q2, "trials": trials,
        "ball_fails": int(ball_fail), "ml_fails": int(ml_fail),
        "log2_ball_empirical": math.log2(ball_fail / trials) if ball_fail else None,
        "log2_ml_empirical": math.log2(ml_fail / trials) if ml_fail else None,
        "log2_ball_predicted": mean_log2(pred_ball),
        "log2_ml_predicted": mean_log2(pred_ml),
    }


def full(samples=40, seed=7):
    """Same ciphertext stream as failure_boost.tail_survey (same seed)."""
    rng = np.random.default_rng(seed)
    model = fb.Model()
    cdf_e = np.cumsum(model.p_e)
    cdf_eps = np.cumsum(model.p_eps)
    rows = []
    for _ in range(samples):
        a = rng.integers(0, 2, size=(fb.ETA_R, fb.N))
        b = rng.integers(0, 2, size=(fb.ETA_R, fb.N))
        r = (a - b).sum(0).astype(np.float64)
        e = model.xs_e[np.searchsorted(cdf_e, rng.random(fb.N))]
        eps = model.xs_eps[np.searchsorted(cdf_eps, rng.random(fb.N))]
        w = np.zeros(fb.N)
        for i in range(fb.N_BLOCKS):
            w[8 * i : 8 * i + 8] = model.codes[int(rng.integers(0, 16))]
        w[fb.HALF :] = w[: fb.HALF]
        v = 2.0 * (e + eps) + w
        Gr = fb.all_block_grams(r)
        Gv = fb.all_block_grams(v)
        lb, lm = [], []
        for blk in range(fb.N_BLOCKS):
            Sig = model.var_g * Gr[blk] + model.var_f * Gv[blk]
            ev = np.clip(np.linalg.eigvalsh(Sig), 1e-9, None)
            lb.append(fb.log2_sf_lams(ev, fb.T_THRESH))
            lm.append(log2_ml_block(Sig, fb.Q))
        rows.append({"log2_ball": fb.log2sumexp2(lb), "log2_ml": fb.log2sumexp2(lm)})
        print(f"{rows[-1]['log2_ball']:8.2f} {rows[-1]['log2_ml']:8.2f}", flush=True)
    lm = np.array([x["log2_ml"] for x in rows])
    lb = np.array([x["log2_ball"] for x in rows])

    # independence model with the ML decoder: Sigma = lambda I per block
    lam = 18399.09
    ind = log2_ml_block(lam * np.eye(16), fb.Q) + math.log2(fb.N_BLOCKS)
    lam_spec = 17165.72
    ind_spec = log2_ml_block(lam_spec * np.eye(16), fb.Q) + math.log2(fb.N_BLOCKS)
    summary = {
        "samples": samples,
        "ball_median": float(np.median(lb)),
        "ball_log2_mean_p": float(np.log2(np.exp2(lb).mean())),
        "ml_mean_log2": float(lm.mean()),
        "ml_std_log2": float(lm.std(ddof=1)),
        "ml_median": float(np.median(lm)),
        "ml_min": float(lm.min()),
        "ml_max": float(lm.max()),
        "ml_log2_mean_p": float(np.log2(np.exp2(lm).mean())),
        "ml_independent_lambda18399": ind,
        "ml_independent_spec_sigma2_17166": ind_spec,
    }
    out = Path(__file__).resolve().parent / "ml_decoder_results.json"
    out.write_text(json.dumps({"summary": summary, "samples": rows}, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    if len(sys.argv) > 1 and sys.argv[1] == "full":
        full()
    else:
        print(json.dumps(toy(), indent=2))
