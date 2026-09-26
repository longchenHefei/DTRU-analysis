"""Cheetah PKE on a scaled-down ring, plus full-size noise checks.

Implements Algorithms 15–17 of the Cheetah specification: ring x^N+1,
Compress(x, D)=floor(x/D), Decompress=min(y D + D/2, q−1),
m' = (2 (v' − Truncate(u' s, λ)) mod ±q) mod 2.
"""

from __future__ import annotations

import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from exact_tails import (  # noqa: E402
    cbd_pmf,
    cheetah_exact_log2,
    cheetah_round_error,
    product_pmf,
    tilted_1d_tail,
)

OUT = os.path.join(HERE, "cheetah_toy.json")


def cbd(rng, shape, eta):
    a = rng.integers(0, 2, size=shape + (eta,))
    b = rng.integers(0, 2, size=shape + (eta,))
    return (a.sum(-1) - b.sum(-1)).astype(np.int64)


def mul_negacyclic(a, b):
    n = len(a)
    tw = np.exp(-1j * np.pi * np.arange(n) / n)
    c = np.fft.ifft(np.fft.fft(a.astype(np.float64) * tw) * np.fft.fft(b.astype(np.float64) * tw))
    return np.round(np.real(c * np.exp(1j * np.pi * np.arange(n) / n))).astype(np.int64)


def compress(x, D, q):
    y = np.floor_divide(x % q, D)
    return np.minimum(y * D + D // 2, q - 1)


def center(x, q):
    t = (x + q // 2) % q - q // 2
    return t


def cheetah_roundtrip(N, k, q, eta, db, du, dv, lam, trials, keys, seed):
    dq = math.ceil(math.log2(q))
    Db = 1 << (dq - db)
    Du = 1 << (dq - du)
    Dv = 1 << (dq - dv)
    Delta = pow(2, -1, q)
    rng = np.random.default_rng(seed)
    dec_fail = pred_fail = disagree = 0
    noises = []
    per = max(1, trials // keys)
    for _ in range(keys):
        A = rng.integers(0, q, size=(k, k, N))
        s = cbd(rng, (k, N), eta)
        e = cbd(rng, (k, N), eta)
        b = np.zeros((k, N), dtype=np.int64)
        for i in range(k):
            for j in range(k):
                b[i] += mul_negacyclic(A[i, j], s[j])
            b[i] = (b[i] + e[i]) % q
        bt = compress(b, Db, q)
        for _t in range(per):
            r = cbd(rng, (k, N), eta)
            e1 = cbd(rng, (k, N), eta)
            e2 = cbd(rng, (N,), eta)
            m = rng.integers(0, 2, size=lam)
            u = np.zeros((k, N), dtype=np.int64)
            for j in range(k):
                for i in range(k):
                    u[j] += mul_negacyclic(r[i], A[i, j])
                u[j] = (u[j] + e1[j]) % q
            v = np.zeros(N, dtype=np.int64)
            for i in range(k):
                v += mul_negacyclic(r[i], bt[i])
            v[:lam] = (v[:lam] + e2[:lam] + Delta * m) % q
            uc = compress(u, Du, q)
            vc = compress(v[:lam], Dv, q)
            su = np.zeros(lam, dtype=np.int64)
            for i in range(k):
                su += mul_negacyclic(uc[i], s[i])[:lam]
            w = center(vc - su, q)
            mp = center(2 * w, q) % 2
            # algebraic noise E = w − Δ m  (first lam coeffs)
            E = center(w - (Delta * m) % q, q)
            noises.append(E.astype(np.float64))
            pred = np.any(np.abs(E) >= q / 4.0)
            real = np.any(mp != m)
            dec_fail += int(real)
            pred_fail += int(pred)
            disagree += int(real != pred)
    noises = np.concatenate(noises)
    emp_var = float(np.mean(noises ** 2))
    # model variance
    sig2 = eta / 2.0
    var_eb = (Db ** 2) / 12.0
    var_eu = (Du ** 2) / 12.0
    var_ev = (Dv ** 2) / 12.0
    model_var = (
        k * N * sig2 * sig2
        + k * N * var_eb * sig2
        + k * N * sig2 * sig2
        + k * N * sig2 * var_eu
        + sig2
        + var_ev
    )
    exact = cheetah_exact_log2(k, N, eta, q, db, du, dv, lam)
    return {
        "params": {
            "N": N, "k": k, "q": q, "eta": eta,
            "db": db, "du": du, "dv": dv, "lam": lam,
            "Db": Db, "Du": Du, "Dv": Dv,
        },
        "keys": keys,
        "encryptions": keys * per,
        "decrypt_failures": dec_fail,
        "predicate_failures": pred_fail,
        "predicate_disagreements": disagree,
        "empirical_failure_log2": (
            None if dec_fail == 0 else math.log2(dec_fail / (keys * per))
        ),
        "empirical_var": emp_var,
        "model_var": model_var,
        "second_moment_ratio": emp_var / model_var,
        "exact_log2": round(exact, 3),
        "emp_std": float(np.std(noises)),
        "emp_max_abs": float(np.max(np.abs(noises))),
    }


def fullsize_noise(name, k, lam, eta, db, du, dv, trials, seed):
    q, N = 7681, 640
    dq = math.ceil(math.log2(q))
    Db = 1 << (dq - db)
    Du = 1 << (dq - du)
    Dv = 1 << (dq - dv)
    Delta = pow(2, -1, q)
    rng = np.random.default_rng(seed)
    Es = []
    fails = 0
    for _ in range(trials):
        A = rng.integers(0, q, size=(k, k, N))
        s = cbd(rng, (k, N), eta)
        e = cbd(rng, (k, N), eta)
        b = np.zeros((k, N), dtype=np.int64)
        for i in range(k):
            for j in range(k):
                b[i] += mul_negacyclic(A[i, j], s[j])
            b[i] = (b[i] + e[i]) % q
        bt = compress(b, Db, q)
        r = cbd(rng, (k, N), eta)
        e1 = cbd(rng, (k, N), eta)
        e2 = cbd(rng, (N,), eta)
        m = rng.integers(0, 2, size=lam)
        u = np.zeros((k, N), dtype=np.int64)
        for j in range(k):
            for i in range(k):
                u[j] += mul_negacyclic(r[i], A[i, j])
            u[j] = (u[j] + e1[j]) % q
        v = np.zeros(N, dtype=np.int64)
        for i in range(k):
            v += mul_negacyclic(r[i], bt[i])
        v[:lam] = (v[:lam] + e2[:lam] + Delta * m) % q
        uc = compress(u, Du, q)
        vc = compress(v[:lam], Dv, q)
        su = np.zeros(lam, dtype=np.int64)
        for i in range(k):
            su += mul_negacyclic(uc[i], s[i])[:lam]
        w = center(vc - su, q)
        E = center(w - (Delta * m) % q, q)
        Es.append(E.astype(np.float64))
        mp = center(2 * w, q) % 2
        fails += int(np.any(mp != m))
    E = np.concatenate(Es)
    sig2 = eta / 2.0
    model_var = (
        k * N * sig2 * sig2
        + k * N * (Db ** 2 / 12.0) * sig2
        + k * N * sig2 * sig2
        + k * N * sig2 * (Du ** 2 / 12.0)
        + sig2
        + (Dv ** 2 / 12.0)
    )
    tails = {}
    for thr in (800, 1000, 1200, q / 4):
        p = float(np.mean(np.abs(E) >= thr))
        tails[str(int(thr))] = None if p == 0 else round(math.log2(p), 3)
    exact = {
        str(int(thr)): round(
            tilted_1d_tail(
                [
                    (product_pmf(cbd_pmf(eta), cbd_pmf(eta)), k * N),
                    (product_pmf(cheetah_round_error(q, db, "b"), cbd_pmf(eta)), k * N),
                    (product_pmf(cbd_pmf(eta), cbd_pmf(eta)), k * N),
                    (product_pmf(cbd_pmf(eta), cheetah_round_error(q, du, "u")), k * N),
                    (cbd_pmf(eta), 1),
                    (cheetah_round_error(q, dv, "v"), 1),
                ],
                thr,
            ),
            3,
        )
        for thr in (800, 1000, 1200)
    }
    return {
        "name": name,
        "trials": trials,
        "decrypt_failures": fails,
        "emp_std": float(np.std(E)),
        "model_std": math.sqrt(model_var),
        "emp_max_abs": float(np.max(np.abs(E))),
        "emp_tail_log2": tails,
        "exact_coeff_tail_log2": exact,
    }


def main():
    out = {}
    print("Cheetah toy (observable failures)", flush=True)
    # N=64, k=2, eta=4, shrink q and compression until failures appear
    out["toy"] = cheetah_roundtrip(
        N=64, k=2, q=769, eta=4, db=8, du=8, dv=5, lam=32,
        trials=20000, keys=40, seed=3,
    )
    print(out["toy"], flush=True)
    print("full-size Cheetah128 noise", flush=True)
    out["full_128"] = fullsize_noise("Cheetah128", 1, 128, 5, 10, 10, 4, 800, seed=5)
    print({k: out["full_128"][k] for k in ("emp_std", "model_std", "emp_tail_log2", "decrypt_failures")}, flush=True)
    print("full-size Cheetah256 noise", flush=True)
    out["full_256"] = fullsize_noise("Cheetah256", 2, 256, 4, 10, 10, 4, 1200, seed=7)
    print({k: out["full_256"][k] for k in ("emp_std", "model_std", "emp_tail_log2", "decrypt_failures")}, flush=True)
    with open(OUT, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
        f.write("\n")
    print("wrote", OUT, flush=True)


if __name__ == "__main__":
    main()
