"""Decryption failure rate of every DTRU parameter set with the real decoder.

Rings
    tri   : x^n - x^{n/2} + 1      p = 2              (648, 768, 1024, 1536, 2048)
    pow2  : x^n + 1                p = 1 - x^{n/2}    (Light)
    lppnf : x^n - x - 1            p = 2              (Prime)

Error polynomial (Theorem 1 of the specification, f = p f' + 1, u = e + eps):
    p = 2          Err = g r + f' (2u + w)              w  = (1 + x^off) w'
    p = 1-x^{n/2}  Err = g r + f' ((1 - x^{n/2}) u + w') + u
and c' f = Err + (q+1)/2 w. Decode block i = octets i and i + off/8.
off = n/2 for even n; for DTRU-Prime (n = 1087) the specification gives no
layout, we take off = floor(n/2) = 543 and nb = floor(n/16) = 67 blocks.

Failure of block i: some codeword difference support D with
    sum_{j in D} |Err_j| >= |D| (q+1)/4     (all sign patterns, ml_decoder_check.DIRS)
Given (r, v) this is a union of half-spaces of a Gaussian with covariance
Sigma = var_g Gram(r) + var_f Gram(v); the top terms are recomputed with the
exact CBD cumulant generating function (exact_cgf_check.log_lr_tail).

    python3 attack/DTRU/dfr_all_sets.py toy          # three rings, real Enc/Dec
    python3 attack/DTRU/dfr_all_sets.py full [SET]   # 40 ciphertexts per set
"""

import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy.special import log_ndtr

import failure_boost as fb
from exact_cgf_check import log_lr_tail
from ml_decoder_check import DIRS, SIZES, decode_block

TOP = 64

SETS = {
    "DTRU-Light": dict(ring="pow2", n=512, q=769, q2=256, eta_g=2, eta_f=1, eta_r=1, eta_e=2,
                       claimed_log2_dfr=-143.92, level=128, ntru_cq=(132, 116), estimator=132.3),
    "DTRU-648": dict(ring="tri", n=648, q=3457, q2=512, eta_g=9, eta_f=2, eta_r=2, eta_e=2,
                     claimed_log2_dfr=-146.79, level=128, ntru_cq=(164, 144), estimator=164.7),
    "DTRU-768": dict(ring="tri", n=768, q=3457, q2=1024, eta_g=4, eta_f=3, eta_r=4, eta_e=2,
                     claimed_log2_dfr=-185.56, level=192, ntru_cq=(195, 171), estimator=199.4),
    "DTRU-1024": dict(ring="tri", n=1024, q=3457, q2=1024, eta_g=5, eta_f=2, eta_r=3, eta_e=2,
                      claimed_log2_dfr=-190.48, level=256, ntru_cq=(270, 237), estimator=277.7),
    "DTRU-Prime": dict(ring="lppnf", n=1087, q=2017, q2=1024, eta_g=2, eta_f=1, eta_r=2, eta_e=1,
                       claimed_log2_dfr=-180.72, level=256, ntru_cq=(280, 246), estimator=292.6),
    "DTRU-1536": dict(ring="tri", n=1536, q=3457, q2=1024, eta_g=3, eta_f=2, eta_r=2, eta_e=1,
                      claimed_log2_dfr=-195.50, level=384, ntru_cq=(416, 365), estimator=416.7),
    "DTRU-2048": dict(ring="tri", n=2048, q=3457, q2=1024, eta_g=5, eta_f=1, eta_r=2, eta_e=1,
                      claimed_log2_dfr=-204.10, level=512, ntru_cq=(571, 501), estimator=512.5),
}


# ---------------------------------------------------------------- ring arithmetic
def mul_by_x(a, ring):
    n = a.shape[0]
    out = np.empty_like(a)
    last = a[-1]
    out[1:] = a[:-1]
    if ring == "tri":
        out[0] = -last
        out[n // 2] += last
    elif ring == "pow2":
        out[0] = -last
    elif ring == "lppnf":
        out[0] = last
        out[1] += last
    else:
        raise ValueError(ring)
    return out


def mul_matrix(poly, ring):
    """Columns poly * x^j, so mul_matrix(a) @ b = a * b in the ring."""
    n = poly.shape[0]
    M = np.empty((n, n), dtype=poly.dtype)
    acc = poly.copy()
    for j in range(n):
        M[:, j] = acc
        acc = mul_by_x(acc, ring)
    return M


def shift(poly, k, ring):
    out = poly.copy()
    for _ in range(k):
        out = mul_by_x(out, ring)
    return out


def layout(n):
    nb = n // 16
    off = n // 2
    return nb, off


def block_idx(b, off):
    return np.concatenate([np.arange(8 * b, 8 * b + 8), np.arange(off + 8 * b, off + 8 * b + 8)])


# ---------------------------------------------------------------- sampling
def sample_pmf(rng, xs, ps, shape):
    return xs[np.searchsorted(np.cumsum(ps), rng.random(shape))]


def sample_cbd(rng, eta, shape):
    a = rng.integers(0, 2, size=(eta,) + shape)
    b = rng.integers(0, 2, size=(eta,) + shape)
    return (a - b).sum(0).astype(np.float64)


def encode_wprime(rng, n, nb, codes):
    """w' : nb random codewords in the first 8*nb coefficients."""
    w = np.zeros(n)
    msg = rng.integers(0, 16, size=nb)
    for b, m in enumerate(msg):
        w[8 * b : 8 * b + 8] = codes[int(m)]
    return w, msg


def noise_terms(P, rng, n, nb, off, codes, xs_e, p_e, xs_eps, p_eps):
    """Return r, u, v, w (full codeword), msg for one honest encryption."""
    ring = P["ring"]
    r = sample_cbd(rng, P["eta_r"], (n,))
    e = sample_cbd(rng, P["eta_e"], (n,))
    eps = sample_pmf(rng, xs_eps, p_eps, (n,))
    u = e + eps
    wp, msg = encode_wprime(rng, n, nb, codes)
    w = wp + shift(wp, off, ring)
    if ring == "pow2":
        v = u - shift(u, n // 2, ring) + wp
    else:
        v = 2.0 * u + w
    return r, u, v, w, msg


def eta_vector(P, n):
    return np.concatenate([np.full(n, float(P["eta_g"])), np.full(n, float(P["eta_f"]))])


# ---------------------------------------------------------------- failure model
def block_terms(Mr16, Mv16, u16, var_g, var_f, q):
    """Gaussian log2 of every (pattern, sign) half-space for one block.

    Returns lg (K x 2) natural-log probabilities, sd (K,), thr (K,), shift (K,).
    """
    Sig = var_g * Mr16 @ Mr16.T + var_f * Mv16 @ Mv16.T
    sd = np.sqrt(np.maximum(np.einsum("ki,ij,kj->k", DIRS, Sig, DIRS), 1e-12))
    thr = SIZES * (q + 1) / 4.0
    sh = DIRS @ u16
    lg = np.stack([log_ndtr(-(thr - sh) / sd), log_ndtr(-(thr + sh) / sd)], axis=1)
    return lg, sd, thr, sh, Sig


def ciphertext_log2(P, r, u, v, top=TOP):
    """(gaussian union, cgf-corrected union, per-block gaussian, diag stats) for one ciphertext."""
    n, q, ring = P["n"], P["q"], P["ring"]
    nb, off = layout(n)
    var_g, var_f = P["eta_g"] / 2.0, P["eta_f"] / 2.0
    Mr, Mv = mul_matrix(r, ring), mul_matrix(v, ring)
    eta = eta_vector(P, n)
    cand = []
    blocks = []
    diag = []
    eig_top = []
    ball = []
    T = (q - 1) ** 2 / 2.0
    for b in range(nb):
        idx = block_idx(b, off)
        lg, sd, thr, sh, Sig = block_terms(Mr[idx], Mv[idx], u[idx], var_g, var_f, q)
        flat = lg.ravel()
        order = np.argsort(flat)[-top:]
        for k in order:
            cand.append((float(flat[k]), b, int(k // 2), int(k % 2)))
        rest = np.delete(flat, order)
        cand.append((float(np.logaddexp.reduce(rest)), b, -1, 0))
        blocks.append(float(np.logaddexp.reduce(flat)) / math.log(2.0))
        diag.append(float(np.trace(Sig) / 16.0))
        ev = np.linalg.eigvalsh(Sig)
        eig_top.append(float(ev[-1]))
        ball.append(fb.log2_sf_lams(np.clip(ev, 1e-9, None), T))
    cand.sort(reverse=True)
    g_terms, x_terms = [], []
    done = 0
    for lg, b, k, sgn in cand:
        g_terms.append(lg)
        if k < 0 or done >= top:
            x_terms.append(lg)
            continue
        done += 1
        idx = block_idx(b, off)
        s = DIRS[k] * (1.0 if sgn == 0 else -1.0)
        a = np.concatenate([Mr[idx].T @ s, Mv[idx].T @ s])
        c = SIZES[k] * (q + 1) / 4.0 - float(s @ u[idx])
        x_terms.append(log_lr_tail(a, eta, c))
    to2 = 1.0 / math.log(2.0)
    return {
        "gauss": float(np.logaddexp.reduce(g_terms)) * to2,
        "cgf": float(np.logaddexp.reduce(x_terms)) * to2,
        "ball_spectrum": fb.log2sumexp2(ball),
        "mean_diag": float(np.mean(diag)),
        "max_eig_over_diag": float(np.max(eig_top) / np.mean(diag)),
        "worst_block_gauss": float(max(blocks)),
    }


# ---------------------------------------------------------------- real pipeline (toy)
def inverse_mod(f, ring, q):
    """f^{-1} in Z_q[x]/(ring) via Gaussian elimination on the multiplication matrix."""
    n = f.shape[0]
    M = mul_matrix(f.astype(np.int64), ring) % q
    A = np.concatenate([M, np.eye(n, dtype=np.int64)], axis=1) % q
    for col in range(n):
        piv = next((r for r in range(col, n) if A[r, col] % q), None)
        if piv is None:
            return None
        A[[col, piv]] = A[[piv, col]]
        inv = pow(int(A[col, col]), -1, q)
        A[col] = (A[col] * inv) % q
        for r in range(n):
            if r != col and A[r, col]:
                A[r] = (A[r] - A[r, col] * A[col]) % q
    return A[:, n:] @ np.eye(n, dtype=np.int64)[:, 0] % q  # f^{-1} * 1 = column 0


def centered(x, q):
    x = np.mod(x, q)
    return np.where(x > q // 2, x - q, x)


def toy(ring, n, q, q2, etas, trials=20000, seed=11, keys=40):
    """Real KeyGen/Enc/Dec at small n over `keys` fresh keys; compare with the
    ball and the half-space model (per-ciphertext Gaussian union, averaged)."""
    rng = np.random.default_rng(seed)
    eta_g, eta_f, eta_r, eta_e = etas
    nb, off = layout(n)
    codes = fb.e8_codewords()
    xs_eps, p_eps = fb.rounding_pmf(q, q2)
    h_half = (q + 1) // 2
    var_g, var_f = eta_g / 2.0, eta_f / 2.0
    T = (q - 1) ** 2 / 2.0
    p_poly = np.zeros(n, dtype=np.int64)
    if ring == "pow2":
        p_poly[0] = 1
        p_poly[n // 2] = -1
    else:
        p_poly[0] = 2

    dec_fail = ball_fail = algebra_mismatch = done = 0
    per_key = []
    pred = []
    sq = np.zeros(n)
    lam_model = np.zeros(n)
    n_model = 0
    for _k in range(keys):
        while True:
            g = sample_cbd(rng, eta_g, (n,)).astype(np.int64)
            fp = sample_cbd(rng, eta_f, (n,)).astype(np.int64)
            f = (mul_matrix(p_poly, ring) @ fp) % q
            f[0] = (f[0] + 1) % q
            finv = inverse_mod(f, ring, q)
            if finv is not None:
                break
        h = (mul_matrix(g % q, ring) @ finv) % q
        Mh, Mf = mul_matrix(h, ring), mul_matrix(f, ring)
        Mg, Mfp = mul_matrix(g.astype(np.float64), ring), mul_matrix(fp.astype(np.float64), ring)
        kf = 0
        for t_i in range(trials // keys):
            r = sample_cbd(rng, eta_r, (n,)).astype(np.int64)
            e = sample_cbd(rng, eta_e, (n,)).astype(np.int64)
            wp, msg = encode_wprime(rng, n, nb, codes)
            wp = wp.astype(np.int64)
            w = wp + shift(wp, off, ring)
            x = (Mh @ r + e + h_half * w) % q
            c = ((2 * x * q2 + q) // (2 * q)) % q2
            cp = ((2 * c * q + q2) // (2 * q2)) % q
            eps = centered(cp - x, q)
            t = (Mf @ cp) % q
            u = (e + eps).astype(np.float64)
            if ring == "pow2":
                v = u - shift(u, n // 2, ring) + wp
            else:
                v = 2.0 * u + w
            err = Mg @ r + Mfp @ v + u
            t_alg = np.mod(np.rint(err).astype(np.int64) + h_half * w, q)
            algebra_mismatch += int(np.any(t_alg != t))
            sq += err * err
            bad_dec = bad_ball = False
            for b in range(nb):
                idx = block_idx(b, off)
                if decode_block(t[idx].astype(np.float64), q) != int(msg[b]):
                    bad_dec = True
                if np.dot(err[idx], err[idx]) >= T:
                    bad_ball = True
            dec_fail += bad_dec
            kf += bad_dec
            ball_fail += bad_ball
            done += 1
            if t_i % 10 == 0:
                Mr, Mv = mul_matrix(r.astype(np.float64), ring), mul_matrix(v, ring)
                logs = []
                for b in range(nb):
                    idx = block_idx(b, off)
                    lg, sd, thr, sh, Sig = block_terms(Mr[idx], Mv[idx], u[idx], var_g, var_f, q)
                    logs.append(float(np.logaddexp.reduce(lg.ravel())) / math.log(2.0))
                    lam_model[idx] += np.diag(Sig)
                n_model += 1
                pred.append(fb.log2sumexp2(logs))
        per_key.append(kf)
    pred = np.array(pred)
    emp_var = sq / done
    mod_var = lam_model / n_model
    return {
        "ring": ring, "n": n, "q": q, "q2": q2, "etas": list(etas), "trials": done, "keys": keys,
        "algebra_mismatch": algebra_mismatch,
        "decoder_fails": dec_fail,
        "ball_fails": ball_fail,
        "fails_per_key_min_max": [int(min(per_key)), int(max(per_key))],
        "log2_decoder": math.log2(dec_fail / done) if dec_fail else None,
        "log2_ball": math.log2(ball_fail / done) if ball_fail else None,
        "log2_model_gauss_union_mean_p": float(np.log2(np.mean(np.exp2(np.minimum(pred, 0.0))))),
        "log2_model_gauss_union_median": float(np.median(pred)),
        "second_moment_ratio_mean": float(np.mean(emp_var[mod_var > 0] / mod_var[mod_var > 0])),
    }


def run_toys():
    out = {}
    out["tri"] = toy("tri", 64, 193, 64, (5, 1, 2, 1))
    out["pow2"] = toy("pow2", 64, 89, 32, (2, 1, 1, 2))
    out["lppnf"] = toy("lppnf", 61, 149, 64, (2, 1, 2, 1))
    return out


# ---------------------------------------------------------------- full sets
def implied_sigma2(claimed_log2, nb, T):
    """sigma^2 that reproduces the claimed delta with nb * P(chi2_16 >= T/sigma^2)."""
    from scipy.optimize import brentq
    target = claimed_log2 - math.log2(nb)
    f = lambda x: fb.chi2_log2_sf(16, x) - target
    x = brentq(f, 50.0, 2000.0)
    return T / x, x


def full(name, samples=40, seed=7):
    P = SETS[name]
    n, q, q2, ring = P["n"], P["q"], P["q2"], P["ring"]
    nb, off = layout(n)
    rng = np.random.default_rng(seed)
    codes = fb.e8_codewords()
    xs_e, p_e = fb.cbd_pmf(P["eta_e"])
    xs_eps, p_eps = fb.rounding_pmf(q, q2)
    T = (q - 1) ** 2 / 2.0
    rows = []
    for i in range(samples):
        r, u, v, w, msg = noise_terms(P, rng, n, nb, off, codes, xs_e, p_e, xs_eps, p_eps)
        row = ciphertext_log2(P, r, u, v)
        rows.append(row)
        print(f"{name} {i:2d} gauss {row['gauss']:8.2f} cgf {row['cgf']:8.2f} "
              f"ball {row['ball_spectrum']:8.2f} eig/diag {row['max_eig_over_diag']:5.2f}", flush=True)
    G = np.array([x["gauss"] for x in rows])
    X = np.array([x["cgf"] for x in rows])
    B = np.array([x["ball_spectrum"] for x in rows])
    lam = float(np.mean([x["mean_diag"] for x in rows]))
    ind = math.log2(nb) + fb.chi2_log2_sf(16, T / lam)
    s2, x_claim = implied_sigma2(P["claimed_log2_dfr"], nb, T)
    ps = np.sort(np.exp2(X))
    boost = []
    for k in (samples, samples // 4, 1):
        top = ps[-k:]
        alpha, beta = k / samples, float(top.mean())
        boost.append({"keep": k, "precomp_bits": -math.log2(alpha), "query_bits": -math.log2(beta),
                      "classical_bits": -math.log2(alpha * beta),
                      "quantum_bits": -0.5 * math.log2(alpha) - math.log2(beta)})
    summary = {
        "set": name, "ring": ring, "n": n, "q": q, "q2": q2, "blocks": nb, "offset": off,
        "etas_g_f_r_e": [P["eta_g"], P["eta_f"], P["eta_r"], P["eta_e"]],
        "claimed_log2_dfr": P["claimed_log2_dfr"], "level": P["level"],
        "ntru_core_svp_claimed": list(P["ntru_cq"]), "core_svp_estimator": P["estimator"],
        "samples": samples,
        "mean_marginal_variance": lam,
        "implied_sigma2_from_claim": s2,
        "independent_chi2_ball_log2": ind,
        "spectrum_ball_median_log2": float(np.median(B)),
        "spectrum_ball_log2_mean_p": float(np.log2(np.mean(np.exp2(B)))),
        "gauss_median_log2": float(np.median(G)),
        "gauss_log2_mean_p": float(np.log2(np.mean(np.exp2(G)))),
        "cgf_median_log2": float(np.median(X)),
        "cgf_mean_log2": float(X.mean()),
        "cgf_std_log2": float(X.std(ddof=1)),
        "cgf_min_log2": float(X.min()),
        "cgf_max_log2": float(X.max()),
        "cgf_log2_mean_p": float(np.log2(np.mean(np.exp2(X)))),
        "max_eig_over_diag_median": float(np.median([x["max_eig_over_diag"] for x in rows])),
        "boosting": boost,
    }
    return summary, rows


if __name__ == "__main__":
    here = Path(__file__).resolve().parent
    cmd = sys.argv[1] if len(sys.argv) > 1 else "toy"
    if cmd == "toy":
        res = run_toys()
        (here / "dfr_toys.json").write_text(json.dumps(res, indent=2))
        print(json.dumps(res, indent=2))
    elif cmd == "full":
        import os
        names = sys.argv[2:] or list(SETS)
        samples = int(os.environ.get("DTRU_SAMPLES", "40"))
        for name in names:
            s, rows = full(name, samples=samples)
            (here / f"dfr_set_{name}.json").write_text(
                json.dumps({"summary": s, "samples": rows}, indent=2))
            print(json.dumps(s, indent=2), flush=True)
    elif cmd == "merge":
        data = {}
        for name in SETS:
            p = here / f"dfr_set_{name}.json"
            if p.exists():
                data[name] = json.loads(p.read_text())
                p.unlink()
        (here / "dfr_all_sets.json").write_text(json.dumps(data, indent=2))
        for name, d in data.items():
            s = d["summary"]
            print(f"{name:11s} claimed {s['claimed_log2_dfr']:8.2f} indep {s['independent_chi2_ball_log2']:8.2f} "
                  f"ball {s['spectrum_ball_log2_mean_p']:8.2f} cgf median {s['cgf_median_log2']:8.2f} "
                  f"cgf mean {s['cgf_log2_mean_p']:8.2f}")
    else:
        raise SystemExit(cmd)
