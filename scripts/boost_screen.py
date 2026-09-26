"""Failure-boosting screen for Scabbard, Rudraksh2, ZEN and NEV.

Scabbard / Rudraksh2: B2-Minal decoder. The absolute failure rate is anchored
to the designers' published delta, because they compute it by an FFT
convolution we do not repeat. The ciphertext-to-ciphertext spread comes from
the squared norm of the encapsulation secret, whose fraction of the noise
variance is computed from the rounding / CBD widths. The cost of a large norm
is the Chernoff rate of that sum of squares.

ZEN: the designers already ran the COSIC failure-boosting scripts at a 2^80
query cap (their Table 2). This file only checks that arithmetic.

NEV: section 8.5 reduces one decryption failure, under a 2^80 query cap, to a
Chernoff bound on the block norm ||v_{i,y}||^2. Sampling that block norm
reproduces their Table 7 means, and the same bound is re-evaluated at query
caps 2^64 and 2^80.
"""

import json
import math
import os
from math import lgamma, log, sqrt

import numpy as np

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "boost_screen_results.json")


def log_binom(n, k):
    if k < 0 or k > n:
        return -math.inf
    return lgamma(n + 1) - lgamma(k + 1) - lgamma(n - k + 1)


def logsumexp(xs):
    xs = [x for x in xs if x > -math.inf]
    if not xs:
        return -math.inf
    m = max(xs)
    return m + log(sum(math.exp(x - m) for x in xs))


def log2sumexp(xs):
    xs = [x for x in xs if x > -math.inf]
    if not xs:
        return -math.inf
    m = max(xs)
    return m + log(sum(math.pow(2.0, x - m) for x in xs)) / log(2)


# ---------------------------------------------------------------- Minal B2


def _cent(x, mod):
    x %= mod
    if x >= mod / 2:
        x -= mod
    return int(x)


def minal_encode(m, q, beta):
    q2, q4 = q // 2, q // 4
    m0, m1, m2, m3 = m
    c0 = (q2 * m0 + q4 * m2 + beta * m3) % q
    c1 = (q2 * m1 + beta * m2 + q4 * m3) % q
    return c0, c1


def minal_decode(c0, c1, q, beta):
    """Phase-2 bits are the sign of being at least q/4 from the low codeword.

    The extracted pseudocode writes floor(|delta| / (q/4)), which equals 2 on
    the exact codeword q/2. A bit is 0 or 1, and the exact codeword must
    decode, so the test is |delta| >= q/4.
    """
    q2, q4 = q // 2, q // 4
    cp0, cp1 = c0 % q2, c1 % q2
    best, m2, m3, bx, by = 10**18, 0, 0, 0, 0
    for a, b in ((0, 0), (0, 1), (1, 0), (1, 1)):
        gx = (q4 * a + beta * b) % q2
        gy = (beta * a + q4 * b) % q2
        dx, dy = _cent(cp0 - gx, q2), _cent(cp1 - gy, q2)
        dist = dx * dx + dy * dy
        if dist < best:
            best, m2, m3, bx, by = dist, a, b, gx, gy
    dx, dy = _cent(c0 - bx, q), _cent(c1 - by, q)
    m0 = 1 if abs(dx) >= q4 else 0
    m1 = 1 if abs(dy) >= q4 else 0
    return (m0, m1, m2, m3)


def failure_radii(q, beta, n_ang=360):
    """Radius at which isotropic noise first leaves the correct cell, per angle.

    For iid gaussian coordinates the radius and the angle are independent, so
    the pair-failure probability is the average of the Rayleigh tails.
    """
    messages = [(a, b, c, d) for a in (0, 1) for b in (0, 1) for c in (0, 1) for d in (0, 1)]
    ang = np.linspace(0, 2 * math.pi, n_ang, endpoint=False)
    dirs = np.stack([np.cos(ang), np.sin(ang)], axis=1)
    out = []
    for m in messages:
        c0, c1 = minal_encode(m, q, beta)
        radii = []
        for v in dirs:
            lo, hi = 0.0, q / 2
            for _ in range(22):
                mid = (lo + hi) / 2
                n0, n1 = int(round(mid * v[0])), int(round(mid * v[1]))
                if minal_decode((c0 + n0) % q, (c1 + n1) % q, q, beta) == m:
                    lo = mid
                else:
                    hi = mid
            radii.append(lo)
        out.append(np.array(radii))
    return out


def log2_pair(radii, sigma):
    """log2 of the average, over messages and angles, of the Rayleigh tail."""
    acc = []
    two_s2 = 2 * sigma * sigma
    for r in radii:
        # mean_angle exp(-r^2 / (2 sigma^2))
        terms = -(r * r) / two_s2 / log(2)
        acc.append(log2sumexp(terms.tolist()) - log(len(r)) / log(2))
    return log2sumexp(acc) - log(len(acc)) / log(2)


def sigma_for_delta(radii, n_pairs, log2_delta):
    """sigma such that n_pairs * p_pair = 2^{log2_delta}."""
    target = log2_delta - log(n_pairs) / log(2)
    lo, hi = 1.0, max(float(np.min(radii[0])) * 2, 10.0)
    # expand until the pair is at least this likely
    for _ in range(30):
        if log2_pair(radii, hi) >= target:
            break
        hi *= 1.6
    for _ in range(50):
        mid = sqrt(lo * hi)
        if log2_pair(radii, mid) < target:
            lo = mid
        else:
            hi = mid
    return hi


# ---------------------------------------------------------------- variance split


def scabbard_split(ell, n, eq, ep, et, eta):
    """Fraction of per-coefficient variance that scales with ||s'||^2.

    Two rounding inner products (keygen error against the ciphertext secret,
    ciphertext rounding against the long-term secret) plus the message
    compression. Only the first scales with the encapsulation secret.
    """
    width = 1 << (eq - ep)
    var_round = (width * width - 1) / 12
    var_s = eta / 2
    drop = eq - et - 2
    comp_width = 1 << drop
    var_comp = (comp_width * comp_width - 1) / 12
    inner = ell * n * var_round * var_s
    total = 2 * inner + var_comp
    return inner / total, total


def rudraksh_split(ell, n, q, p, t, eta):
    """Same split for MLWE plus two compressions.

    The encapsulation seed draws s', e' and e''. Both inner products scale
    when that seed is chosen with a large squared norm. Compression noise does
    not.
    """
    var = eta / 2.0
    inner = ell * n * var * var
    var_p = (q / p) ** 2 / 12
    var_t = (q / t) ** 2 / 12
    comp_p = ell * n * var_p * var
    total = 2 * inner + var + comp_p + var_t
    return (2 * inner) / total, total


def cbd_square_probs(eta):
    """P(s^2 = k) for one CBD(eta) coefficient."""
    counts = {}
    total = 1 << (2 * eta)
    for mask in range(total):
        s = 0
        for i in range(eta):
            s += (mask >> i) & 1
            s -= (mask >> (eta + i)) & 1
        counts[s * s] = counts.get(s * s, 0) + 1
    return {k: v / total for k, v in counts.items()}


def log_pmf_sum_squares(eta, n_coeffs):
    """log P(sum of n_coeffs independent CBD squares = t), t = 0,1,...,eta^2 * n."""
    probs = cbd_square_probs(eta)
    # values are 0,1,...,eta^2 but only some have mass. Enumerate the count of
    # each positive square via one multinomial if few atoms.
    atoms = sorted(k for k in probs if k > 0)
    max_sq = eta * eta
    logs = [-math.inf] * (max_sq * n_coeffs + 1)
    if len(atoms) == 1:
        v = atoms[0]
        p = probs[v]
        p0 = probs.get(0, 0.0)
        for k in range(n_coeffs + 1):
            lp = log_binom(n_coeffs, k) + k * log(p) + (n_coeffs - k) * log(p0)
            logs[k * v] = lp
        return logs
    # two positive atoms (CBD eta=2: 1 and 4) or more: enumerate the rarer one
    rare = max(atoms)
    common = [a for a in atoms if a != rare]
    assert len(common) == 1
    v_c, v_r = common[0], rare
    p_c, p_r, p0 = probs[v_c], probs[v_r], probs.get(0, 0.0)
    for n_r in range(n_coeffs + 1):
        rest = n_coeffs - n_r
        base = log_binom(n_coeffs, n_r) + n_r * log(p_r)
        for n_c in range(rest + 1):
            lp = base + log_binom(rest, n_c) + n_c * log(p_c) + (rest - n_c) * log(p0)
            logs[n_c * v_c + n_r * v_r] = logsumexp([logs[n_c * v_c + n_r * v_r], lp])
    return logs


def chernoff_rate_sq(eta, mean_sq):
    """Rate function of one CBD square, nats, for E[s^2] tilted to mean_sq."""
    probs = cbd_square_probs(eta)
    atoms = sorted(probs)
    hi = max(atoms)
    if mean_sq <= sum(k * probs[k] for k in atoms) + 1e-12:
        return 0.0
    if mean_sq >= hi - 1e-9:
        return math.inf
    best = 0.0
    for theta in np.linspace(0.0, 8.0, 400):
        mgf = sum(probs[k] * math.exp(theta * k) for k in atoms)
        rate = theta * mean_sq - log(mgf)
        if rate > best:
            best = rate
    return best


# ---------------------------------------------------------------- one parameter set


def screen_minal(name, q, beta, n, claimed_log2, w, var_model, eta, n_controlled, level):
    radii = failure_radii(q, beta)
    n_pairs = n // 2
    sigma = sigma_for_delta(radii, n_pairs, claimed_log2)
    # unanchored gaussian model at the same geometry
    sigma_raw = sqrt(var_model)
    raw = log2_pair(radii, sigma_raw) + log(n_pairs) / log(2)

    mean_sq = sum(k * p for k, p in cbd_square_probs(eta).items())
    max_sq = max(cbd_square_probs(eta))
    max_scale = max_sq / mean_sq

    def budgets_at(sigma0):
        def log2_p(scale):
            sig = sigma0 * sqrt((1 - w) + w * scale)
            lp = log2_pair(radii, sig) + log(n_pairs) / log(2)
            return min(lp, 0.0)

        def scale_for(target_log2):
            if log2_p(max_scale) < target_log2:
                return None
            lo, hi = 1.0, max_scale
            for _ in range(60):
                mid = (lo + hi) / 2
                if log2_p(mid) < target_log2:
                    lo = mid
                else:
                    hi = mid
            return hi

        budgets = {}
        for cap in (64, 80):
            sc = scale_for(-cap)
            if sc is None:
                budgets[str(cap)] = {
                    "reachable": False,
                    "best_log2_p": log2_p(max_scale),
                    "precomp_bits": None,
                    "oracle_bits": None,
                    "classical_bits": None,
                    "quantum_bits": None,
                }
                continue
            rate = chernoff_rate_sq(eta, sc * mean_sq)
            pre = n_controlled * rate / log(2)
            budgets[str(cap)] = {
                "reachable": True,
                "scale": sc,
                "precomp_bits": pre,
                "oracle_bits": float(cap),
                "classical_bits": pre + cap,
                "quantum_bits": 0.5 * pre + cap,
            }
        return budgets, log2_p(1.0), log2_p(max_scale)

    budgets, p_mean, p_max = budgets_at(sigma)
    raw_budgets, _, _ = budgets_at(sigma_raw)
    return {
        "name": name,
        "level": level,
        "claimed_log2_dfr": claimed_log2,
        "model_log2_dfr": raw,
        "variance_fraction_controlled": w,
        "anchored_sigma": sigma,
        "model_sigma": sigma_raw,
        "min_radius": float(min(np.min(r) for r in radii)),
        "log2_p_at_mean": p_mean,
        "log2_p_at_max_secret": p_max,
        "budgets": budgets,
        "model_budgets": raw_budgets,
    }


# ---------------------------------------------------------------- NEV block norm


def mul_yk(a, b, k=4):
    out = np.zeros(k, dtype=np.int64)
    for i in range(k):
        for j in range(k):
            t = i + j
            if t < k:
                out[t] += int(a[i]) * int(b[j])
            else:
                out[t - k] -= int(a[i]) * int(b[j])
    return out


def _draw_block(kind, rng, k=4):
    if kind[0] == "B":
        eta = kind[1]
        a = rng.integers(0, 2, size=(eta, k))
        b = rng.integers(0, 2, size=(eta, k))
        return a.sum(0) - b.sum(0)
    sig = kind[1]
    u = rng.random(k)
    x = np.zeros(k, dtype=np.int64)
    x[u < sig] = 1
    x[(u >= sig) & (u < 2 * sig)] = -1
    return x


def nev_block_samples(r_kind, e_kind, scale_r, n_samples, seed):
    rng = np.random.default_rng(seed)
    y = np.ones(4, dtype=np.int64)
    xs = np.empty(n_samples)
    for i in range(n_samples):
        r = _draw_block(r_kind, rng)
        e = _draw_block(e_kind, rng)
        vr, ve = mul_yk(r, y), mul_yk(e, y)
        xs[i] = scale_r * float(vr @ vr) + float(ve @ ve)
    return xs


def nev_chernoff_bits(xs, target, n_blocks):
    best, best_th = 0.0, None
    for theta in np.linspace(0.002, 1.5, 120):
        z = theta * xs
        if z.max() > 700:
            continue
        log_m = log(np.mean(np.exp(z)))
        rate = theta * target - log_m
        if rate > best:
            best, best_th = rate, float(theta)
    return n_blocks * best / log(2), best_th


def screen_nev(name, n, kappa, mu_pub, r_kind, e_kind, scale_r, claimed_cost, samples=40000):
    k = 4
    n_blocks = n // k
    xs = nev_block_samples(r_kind, e_kind, scale_r, samples, seed=hash(name) % 10**6)
    mu = float(xs.mean())
    out = {"name": name, "mu_sampled": mu, "mu_published": mu_pub, "published_classical": claimed_cost}
    for cap in (64, 80):
        bits = cap + math.log2(n_blocks) + k
        c = kappa / sqrt(2 * log(2) * bits)
        target = c * c * mu
        pre, theta = nev_chernoff_bits(xs, target, n_blocks)
        out[str(cap)] = {
            "c": c,
            "theta": theta,
            "precomp_bits": pre,
            "oracle_bits": cap + 0.0,  # their union bound is built to equal the cap
            "classical_bits": cap + pre,
            "quantum_bits": cap + 0.5 * pre,
        }
    return out


# ---------------------------------------------------------------- ZEN arithmetic


ZEN_TABLE2 = [
    # (name, log2 DFR, log2 alpha, classical, quantum) at query cap 2^80
    ("ZEN-light 769->128", -135.1, -78.8, 158.8, 119.4),
    ("ZEN-128 769->256", -128.3, -69.1, 149.1, 114.5),
    ("ZEN-256 769->256", -175.1, -456.2, 536.2, 308.1),
    ("ZEN-256 backup 769->769", -256.3, -2000.5, 2080.5, 1080.2),
    ("ZEN-512 769->256", -179.6, -668.4, 748.4, 414.2),
    ("ZEN-512 backup 769->769", -339.6, -3661.3, 3741.3, 1910.6),
]


def screen_zen():
    rows = []
    for name, dfr, log_alpha, classical, quantum in ZEN_TABLE2:
        # their formula: classical = 80 - log2(alpha), quantum = 80 - 0.5 log2(alpha)
        rows.append({
            "name": name,
            "log2_dfr": dfr,
            "log2_alpha": log_alpha,
            "classical_check": 80 - log_alpha,
            "quantum_check": 80 - 0.5 * log_alpha,
            "published_classical": classical,
            "published_quantum": quantum,
        })
    return rows


# ---------------------------------------------------------------- parameter tables


def run_screen():
    minal = []
    # Scabbard Table 1. eta=2, B=2. Claimed deltas from section 3.2.6.
    scabbard = [
        ("Scabbard-128", 9, 64, 14, 10, 3, 2**14, 1030, -99, 128),
        ("Scabbard-256", 9, 128, 13, 11, 2, 2**13, 0, -125, 256),
        ("Scabbard-512", 8, 256, 13, 11, 6, 2**13, 550, -125, 512),
    ]
    for name, ell, n, eq, ep, et, q, beta, delta, level in scabbard:
        w, var = scabbard_split(ell, n, eq, ep, et, 2)
        minal.append(screen_minal(name, q, beta, n, delta, w, var, 2, ell * n, level))

    # Rudraksh2 Table 1. p and t are powers of two. (B, beta) with B=2.
    rudraksh = [
        ("Rudraksh2-128-I", 9, 64, 3329, 2**12, 2**6, 2, 220, -100, 128),
        ("Rudraksh2-128-II", 9, 64, 4001, 2**9, 2**7, 1, 270, -100, 128),
        ("Rudraksh2-256-I", 9, 128, 3329, 2**11, 2**9, 1, 220, -161, 256),
        ("Rudraksh2-256-II", 9, 128, 4001, 2**11, 2**5, 1, 250, -167, 256),
        ("Rudraksh2-512-I", 8, 256, 7681, 2**13, 2**7, 2, 510, -160, 512),
        ("Rudraksh2-512-II", 9, 256, 4001, 2**12, 2**8, 1, 270, -160, 512),
    ]
    for name, ell, n, q, p, t, eta, beta, delta, level in rudraksh:
        w, var = rudraksh_split(ell, n, q, p, t, eta)
        # s' and e' are both drawn from the seed: 2*ell*n controlled squares
        minal.append(screen_minal(name, q, beta, n, delta, w, var, eta, 2 * ell * n, level))

    # NEV Table 7. scale_r = Var(chi_g) / Var(chi_f). T_{1/3} is uniform ternary
    # (P=1/3 each), which the sampler encodes as sigma=1/3. T_{1/8} has P(±1)=1/8.
    # kappa from Table 7. n from the public-key size.
    B = lambda eta: ("B", eta)
    T = lambda sig: ("T", sig)
    nev_rows = [
        ("NEV-C1", 512, 15.35, 37.47, T(1 / 3), B(2), 1.0 / 0.5, 288),
        ("NEV-R1", 512, 14.18, 48.0, B(3), B(3), 1.0, 151),
        ("NEV-D1", 512, 14.28, 112.0, B(7), B(7), 1.0, 149),
        ("NEV-C2", 1024, 15.49, 21.37, T(1 / 3), B(2), 0.25 / 0.5, 324),
        ("NEV-R2", 1024, 15.35, 32.0, B(2), B(2), 1.0, 369),
        ("NEV-D2", 1024, 17.82, 64.0, B(4), B(4), 1.0, 707),
        ("NEV-C3", 2048, 15.58, 18.74, T(1 / 3), B(1), 1.0, 1098),
        ("NEV-R3", 2048, 14.89, 26.67, T(1 / 3), B(2), 1.0, 542),
        ("NEV-D3", 2048, 20.83, 48.0, B(2), B(3), 1.5 / 1.0, 2994),
    ]
    nev = [screen_nev(*row) for row in nev_rows]
    zen = screen_zen()
    return {"minal": minal, "nev": nev, "zen": zen}


# ---------------------------------------------------------------- toys


def _negacyclic_mul(a, b, q):
    n = len(a)
    out = np.zeros(n, dtype=np.int64)
    for i in range(n):
        for j in range(n):
            t = i + j
            if t < n:
                out[t] += int(a[i]) * int(b[j])
            else:
                out[t - n] -= int(a[i]) * int(b[j])
    return out % q


def _cbd_poly(n, eta, rng):
    a = rng.integers(0, 2, size=(eta, n))
    b = rng.integers(0, 2, size=(eta, n))
    return a.sum(0) - b.sum(0)


def _mat_vec(A, s, q):
    ell, n = len(s), len(s[0])
    out = []
    for i in range(ell):
        acc = np.zeros(n, dtype=np.int64)
        for j in range(ell):
            acc += _negacyclic_mul(A[i][j], s[j], q)
        out.append(acc % q)
    return out


def toy_scabbard(trials=2000, keys=8, seed=1):
    """Small power-of-two MLWR with the same decoder. Failures are visible."""
    ell, n, eta = 2, 16, 2
    eq, ep, et = 8, 5, 3
    q, p = 1 << eq, 1 << ep
    beta = 12
    d = eq - ep
    h = 1 << (d - 1)
    drop = eq - et - 2
    rng = np.random.default_rng(seed)
    noise = []
    dec_fail = 0
    pred_fail = 0
    disagree = 0
    for _ in range(keys):
        A = rng.integers(0, q, size=(ell, ell, n))
        s = [_cbd_poly(n, eta, rng) % q for _ in range(ell)]
        b = []
        for i in range(ell):
            acc = np.zeros(n, dtype=np.int64)
            for j in range(ell):
                acc += _negacyclic_mul(A[j][i], s[j], q)  # A^T
            b.append(((acc + h) % q) >> d)
        for _t in range(trials // keys):
            sp = [_cbd_poly(n, eta, rng) % q for _ in range(ell)]
            u_acc = _mat_vec(A, sp, q)
            u = [((row + h) % q) >> d for row in u_acc]
            # v' = b^T s' mod p
            vp = np.zeros(n, dtype=np.int64)
            for j in range(ell):
                vp += _negacyclic_mul(b[j], sp[j], p)
            vp %= p
            msg = rng.integers(0, 2, size=2 * n)
            mu = np.zeros(n, dtype=np.int64)
            for i in range(n // 2):
                mu[i], mu[i + n // 2] = minal_encode(tuple(int(x) for x in msg[4 * i:4 * i + 4]), q, beta)
            v = ((vp.astype(np.int64) << (eq - ep)) - mu) % q
            v = v >> drop
            # decrypt
            w = np.zeros(n, dtype=np.int64)
            for j in range(ell):
                w += _negacyclic_mul(u[j], s[j], p)
            w %= p
            mprime = ((w.astype(np.int64) << (eq - ep)) - (v.astype(np.int64) << drop)) % q
            got = []
            pred = []
            for i in range(n // 2):
                dec = minal_decode(int(mprime[i]), int(mprime[i + n // 2]), q, beta)
                got.extend(dec)
                # noise = decoded input minus the codeword
                c0, c1 = int(mu[i]), int(mu[i + n // 2])
                e0 = _cent(int(mprime[i]) - c0, q)
                e1 = _cent(int(mprime[i + n // 2]) - c1, q)
                noise.append(e0 * e0 + e1 * e1)
                pred.extend(minal_decode((c0 + e0) % q, (c1 + e1) % q, q, beta))
            if got != list(msg):
                dec_fail += 1
            if pred != list(msg):
                pred_fail += 1
            if (got == list(msg)) != (pred == list(msg)):
                disagree += 1
    w, var = scabbard_split(ell, n, eq, ep, et, eta)
    # each stored value is e0^2+e1^2, expectation 2*var per pair if the model holds
    emp = float(np.mean(noise)) / 2
    return {
        "scheme": "Scabbard-toy",
        "params": {"ell": ell, "n": n, "q": q, "p": p, "t": 1 << et, "eta": eta, "beta": beta},
        "trials": trials,
        "decrypt_failures": dec_fail,
        "predicate_failures": pred_fail,
        "predicate_disagreements": disagree,
        "empirical_coeff_var": emp,
        "model_coeff_var": var,
        "second_moment_ratio": emp / var,
        "controlled_fraction": w,
    }


def toy_rudraksh(trials=2000, keys=8, seed=2):
    ell, n, eta = 2, 16, 1
    q, p, t = 97, 32, 16
    beta = 8
    rng = np.random.default_rng(seed)
    noise_sq = []
    dec_fail = pred_fail = disagree = 0

    def compress(x, mod):
        return int(round(mod / q * (int(x) % q))) % mod

    def decompress(x, mod):
        return int(round(q / mod * int(x))) % q

    per = trials // keys
    for _ in range(keys):
        A = rng.integers(0, q, size=(ell, ell, n))
        s = [_cbd_poly(n, eta, rng) % q for _ in range(ell)]
        b = []
        for i in range(ell):
            acc = np.zeros(n, dtype=np.int64)
            for j in range(ell):
                acc += _negacyclic_mul(A[i][j], s[j], q)
            eb = _cbd_poly(n, eta, rng)
            b.append((acc + eb) % q)
        for _t in range(per):
            sp = [_cbd_poly(n, eta, rng) % q for _ in range(ell)]
            ep = [_cbd_poly(n, eta, rng) % q for _ in range(ell)]
            epp = _cbd_poly(n, eta, rng)
            bp = []
            for i in range(ell):
                acc = np.zeros(n, dtype=np.int64)
                for j in range(ell):
                    acc += _negacyclic_mul(A[j][i], sp[j], q)  # A^T s'
                bp.append((acc + ep[i]) % q)
            cm = np.zeros(n, dtype=np.int64)
            for j in range(ell):
                cm += _negacyclic_mul(b[j], sp[j], q)
            msg = rng.integers(0, 2, size=2 * n)
            mu = np.zeros(n, dtype=np.int64)
            for i in range(n // 2):
                mu[i], mu[i + n // 2] = minal_encode(tuple(int(x) for x in msg[4 * i:4 * i + 4]), q, beta)
            cm = (cm + epp + mu) % q
            u = [compress(x, p) for row in bp for x in row]
            # reshape
            u = [np.array([compress(x, p) for x in row]) for row in bp]
            v = np.array([compress(x, t) for x in cm])
            up = [np.array([decompress(x, p) for x in row]) for row in u]
            vp = np.array([decompress(x, t) for x in v])
            inner = np.zeros(n, dtype=np.int64)
            for j in range(ell):
                inner += _negacyclic_mul(up[j], s[j], q)
            mprime = (vp - inner) % q
            got, pred = [], []
            for i in range(n // 2):
                dec = minal_decode(int(mprime[i]), int(mprime[i + n // 2]), q, beta)
                got.extend(dec)
                e0 = _cent(int(mprime[i]) - int(mu[i]), q)
                e1 = _cent(int(mprime[i + n // 2]) - int(mu[i + n // 2]), q)
                noise_sq.append(e0 * e0 + e1 * e1)
                pred.extend(minal_decode((int(mu[i]) + e0) % q, (int(mu[i + n // 2]) + e1) % q, q, beta))
            if got != list(msg):
                dec_fail += 1
            if pred != list(msg):
                pred_fail += 1
            if (got == list(msg)) != (pred == list(msg)):
                disagree += 1
    w, var = rudraksh_split(ell, n, q, p, t, eta)
    emp = float(np.mean(noise_sq)) / 2
    return {
        "scheme": "Rudraksh2-toy",
        "params": {"ell": ell, "n": n, "q": q, "p": p, "t": t, "eta": eta, "beta": beta},
        "trials": trials,
        "decrypt_failures": dec_fail,
        "predicate_failures": pred_fail,
        "predicate_disagreements": disagree,
        "empirical_coeff_var": emp,
        "model_coeff_var": var,
        "second_moment_ratio": emp / var,
        "controlled_fraction": w,
    }


def _poly_inv(a, q):
    n = len(a)
    M = [list(np.roll(a, j).astype(object)) for j in range(n)]
    # negacyclic: rolling is not enough, signs flip
    M = [[0] * n for _ in range(n)]
    for j in range(n):
        col = _negacyclic_mul(a, np.eye(n, dtype=np.int64)[j], q)
        for i in range(n):
            M[i][j] = int(col[i])
    # augment
    A = [row + [1 if i == j else 0 for j in range(n)] for i, row in enumerate(M)]
    for col in range(n):
        piv = None
        for r in range(col, n):
            if A[r][col] % q != 0:
                piv = r
                break
        if piv is None:
            return None
        A[col], A[piv] = A[piv], A[col]
        inv = pow(int(A[col][col]) % q, -1, q)
        A[col] = [(int(x) * inv) % q for x in A[col]]
        for r in range(n):
            if r == col:
                continue
            f = A[r][col] % q
            if f:
                A[r] = [(int(A[r][c]) - f * int(A[col][c])) % q for c in range(2 * n)]
    return np.array([A[i][n + i] for i in range(n)], dtype=np.int64)  # wrong, want the solution column
    # fix below


def _poly_inv_fix(a, q):
    n = len(a)
    A = [[0] * (2 * n) for _ in range(n)]
    for j in range(n):
        e = np.zeros(n, dtype=np.int64)
        e[j] = 1
        col = _negacyclic_mul(a, e, q)
        for i in range(n):
            A[i][j] = int(col[i])
        A[j][n + j] = 1
    for col in range(n):
        piv = next((r for r in range(col, n) if A[r][col] % q != 0), None)
        if piv is None:
            return None
        A[col], A[piv] = A[piv], A[col]
        inv = pow(int(A[col][col]) % q, -1, q)
        A[col] = [(int(x) * inv) % q for x in A[col]]
        for r in range(n):
            if r == col:
                continue
            f = int(A[r][col]) % q
            if f:
                A[r] = [(int(A[r][c]) - f * int(A[col][c])) % q for c in range(2 * n)]
    # Right-hand block is M^{-1}. Its first column is the inverse polynomial.
    return np.array([A[i][n] for i in range(n)], dtype=np.int64)


def _mul_z(a, b):
    n = len(a)
    out = np.zeros(n, dtype=np.int64)
    for i in range(n):
        for j in range(n):
            t = i + j
            if t < n:
                out[t] += int(a[i]) * int(b[j])
            else:
                out[t - n] -= int(a[i]) * int(b[j])
    return out


def toy_nev(trials=4000, seed=3):
    """Ring product g*r + f'*e against the L1 group test, on a small ring.

    Section 3.2 decrypts correctly when every 4-coefficient block has L1 norm
    below k(q-1)/4. The second moment is taken on the integer product, before
    reduction, so a wrap does not bias it.
    """
    n, k, q, eta = 32, 4, 61, 2
    rng = np.random.default_rng(seed)
    thr = k * (q - 1) / 4
    fails = 0
    sq = []
    pred_mismatch = 0
    for _ in range(trials):
        g = _cbd_poly(n, eta, rng)
        f = _cbd_poly(n, eta, rng)
        r = _cbd_poly(n, eta, rng)
        e = _cbd_poly(n, eta, rng)
        err = _mul_z(g, r) + _mul_z(f, e)
        sq.extend(int(x) for x in err)
        centered = np.array([_cent(int(x), q) for x in err])
        bad = False
        for j in range(n // k):
            idxs = [(j + t * (n // k)) % n for t in range(k)]
            group = centered[idxs]
            l1 = int(np.sum(np.abs(group)))
            worst = max(abs(int(group @ y)) for y in _pm1(k))
            if worst != l1:
                pred_mismatch += 1
            if l1 >= thr:
                bad = True
        if bad:
            fails += 1
    var_model = n * (eta / 2) * (eta / 2) * 2  # gr and f'e
    emp = float(np.mean(np.square(sq)))
    return {
        "scheme": "NEV-toy",
        "params": {"n": n, "k": k, "q": q, "eta": eta, "threshold": thr},
        "trials": trials,
        "l1_failures": fails,
        "l1_dual_mismatches": pred_mismatch,
        "empirical_coeff_var": emp,
        "model_coeff_var": var_model,
        "second_moment_ratio": emp / var_model,
    }


def _pm1(k):
    for mask in range(1 << k):
        yield np.array([1 if (mask >> i) & 1 else -1 for i in range(k)], dtype=np.int64)


def _fast_inv_mod2(f):
    """Algorithm 1, returned in Z2[x] of length len(f). d = n/2 = len(f)."""
    d = len(f)
    if all((int(f[i]) + (1 if i == 0 else 0)) % 2 == 0 for i in range(d)):
        # f ≡ 0 mod (x+1, 2) iff f(1) ≡ 0 mod 2
        pass
    if sum(int(x) for x in f) % 2 == 0:
        return None
    # k = (f+1)/(x+1) over F2. Polynomial division by x+1.
    ff = [int(x) % 2 for x in f]
    ff[0] ^= 1
    k = [0] * (d - 1)
    acc = 0
    for i in range(d - 1):
        acc ^= ff[i]
        k[i] = acc
    finv = [1] + [0] * (d - 1)
    L = int(round(math.log2(d)))
    for step in range(L):
        mod = 1 << step  # x^{2^step} + 1
        # b = k * finv mod (x^{mod}+1, 2), length mod
        b = [0] * mod
        for i, ki in enumerate(k):
            if not ki:
                continue
            for j, fj in enumerate(finv):
                if not fj:
                    continue
                t = (i + j) % (2 * mod)
                if t < mod:
                    b[t] ^= 1
                elif t < 2 * mod:
                    b[t - mod] ^= 1
        # k = k + f*b  mod (x^n+1, 2), then divide by x^{mod}+1
        n = d
        fb = [0] * n
        for i, fi in enumerate(ff):
            # use original f, not f+1. Spec: k <- k + f*b
            pass
        fbits = [int(x) % 2 for x in f]
        fb = [0] * n
        for i, fi in enumerate(fbits):
            if not fi:
                continue
            for j in range(mod):
                if not b[j]:
                    continue
                t = i + j
                if t < n:
                    fb[t] ^= 1
                else:
                    fb[t - n] ^= 1
        k_full = [0] * n
        for i, ki in enumerate(k):
            k_full[i] = ki
        for i in range(n):
            k_full[i] ^= fb[i]
        # divide by x^{mod}+1: spec keeps a quotient of length n/mod? The
        # algorithm is the standard Itoh-style lift and is easy to get wrong
        # from the broken line breaks. Fall back to gaussian elimination.
        return _inv_mod2_gaussian(f)
    return finv


def _inv_mod2_gaussian(f):
    """Inverse of f in F2[x]/(x^{n/2}+1) if the spec's d=n/2, else in x^n+1.

    Decrypt multiplies modulo x^{n/2}+1, so the inverse is modulo that ring.
    """
    n = len(f)
    A = [[0] * (2 * n) for _ in range(n)]
    for j in range(n):
        e = np.zeros(n, dtype=np.int64)
        e[j] = 1
        col = _mul_mod2(f, e)
        for i in range(n):
            A[i][j] = int(col[i])
        A[j][n + j] = 1
    for col in range(n):
        piv = next((r for r in range(col, n) if A[r][col]), None)
        if piv is None:
            return None
        A[col], A[piv] = A[piv], A[col]
        for r in range(n):
            if r != col and A[r][col]:
                for c in range(2 * n):
                    A[r][c] ^= A[col][c]
    return np.array([A[i][n] for i in range(n)], dtype=np.int64)


def _mul_mod2(a, b):
    n = len(a)
    out = np.zeros(n, dtype=np.int64)
    for i in range(n):
        if int(a[i]) % 2 == 0:
            continue
        for j in range(n):
            if int(b[j]) % 2 == 0:
                continue
            t = i + j
            if t < n:
                out[t] ^= 1
            else:
                out[t - n] ^= 1
    return out


def _simple_decoding(a_mod_q, m_prime, finv, q):
    """Algorithm 2. Parity uses the representative in 0..q-1; gaps use the centered lift."""
    n = len(a_mod_q)
    nu = n // 4
    centered = [_cent(int(x), q) for x in a_mod_q]
    # q is odd, so the representative in 0..q-1 flips the bit of a negative
    # centered coefficient. The message polynomial lives in the centered lift.
    parity = [int(x) % 2 for x in centered]
    delta = np.zeros(nu, dtype=np.int64)
    for i in range(nu):
        s = 0
        for t in range(4):
            s ^= parity[(i + t * nu) % n]
        delta[i] = s
    eta = np.zeros(n // 2, dtype=np.int64)
    for i in range(nu):
        def gap(idx):
            return abs(abs(centered[idx]) - q / 2)

        v0 = min(gap(i), gap(i + 2 * nu))
        v1 = min(gap(i + nu), gap(i + 3 * nu))
        if v0 < v1:
            eta[i] ^= delta[i]
        else:
            eta[i + nu] ^= delta[i]
    # m = (m' + eta*finv) mod (2, x^{n/2}+1), then mod (2, x^{n/4})
    prod = _mul_mod2_half(eta, finv)
    rec = (m_prime + prod) % 2
    return rec[:nu]


def _mul_mod2_half(a, b):
    n = len(a)
    out = np.zeros(n, dtype=np.int64)
    for i in range(n):
        if int(a[i]) % 2 == 0:
            continue
        for j in range(n):
            if int(b[j]) % 2 == 0:
                continue
            t = i + j
            if t < n:
                out[t] ^= 1
            else:
                out[t - n] ^= 1
    return out


def toy_zen(trials=200, seed=4):
    """Small ZEN without compression: real keygen, encrypt, decrypt.

    n=32, q=769, secrets uniform in {-1,0,1}. Checks that decrypt matches the
    message, and that the coefficient second moment of g*s+f*e matches n*var^2*2.
    """
    n, q = 32, 769
    rng = np.random.default_rng(seed)
    ok = fail = inv_fail = 0
    sq = []
    half = n // 2

    def small():
        return rng.integers(-1, 2, size=n)

    for _ in range(trials):
        f = small()
        g = small()
        finv_q = _poly_inv_fix(f % q, q)
        # mod-2 inverse on the full ring, then the decrypt reduces mod x^{n/2}+1.
        # Use the full-ring inverse mod 2 of f, reduced: finv such that f*finv=1 mod (x^{n/2}+1, 2)
        # Spec calls FastInversion(f, n/2). Build f mod (x^{n/2}+1, 2).
        f_half = np.array([(int(f[i]) + int(f[i + half])) % 2 for i in range(half)], dtype=np.int64)
        # x^n+1 = (x^{n/2}+1)(x^{n/2}-1) but over F2, x^n+1 = (x^{n/2}+1)^2.
        # The parity of coefficients distance n/2: f mod (x^{n/2}+1) is f[i]+f[i+n/2].
        finv = _inv_mod2_gaussian(f_half)
        if finv_q is None or finv is None:
            inv_fail += 1
            continue
        h = _negacyclic_mul(g % q, finv_q, q)
        s = small()
        e = small()
        m = rng.integers(0, 2, size=n // 4)
        # (1 + x^{n/4} + x^{n/2} + x^{3n/4}) * m, m supported on degree < n/4
        enc = np.zeros(n, dtype=np.int64)
        step = n // 4
        for i, bit in enumerate(m):
            if not bit:
                continue
            for t in range(4):
                enc[(i + t * step) % n] ^= 1
        scale = (q + 1) // 2
        c = (_negacyclic_mul(h, s % q, q) + (e % q) + scale * enc) % q
        fc = _negacyclic_mul(f % q, c, q)
        # (1 + x^{n/2}) * p over x^n+1: out[i] = p[i] - p[i+n/2], out[i+n/2] = p[i] + p[i+n/2].
        a = np.zeros(n, dtype=np.int64)
        a[:half] = (fc[:half] - fc[half:]) % q
        a[half:] = (fc[:half] + fc[half:]) % q
        # Over F2, x^{n/2} = 1 inside x^{n/2}+1, so the half-ring image is the xor of the two halves.
        a_half = np.array([(_cent(int(a[i]), q) + _cent(int(a[i + half]), q)) % 2 for i in range(half)], dtype=np.int64)
        m_prime = _mul_mod2_half(a_half, finv)
        got = _simple_decoding(a, m_prime, finv, q)
        if np.array_equal(got, m):
            ok += 1
        else:
            fail += 1
        err = (_negacyclic_mul(g, s, q) + _negacyclic_mul(f, e, q)) % q
        sq.extend(_cent(int(x), q) for x in err)
    var_one = 2 / 3  # uniform {-1,0,1}
    model = n * var_one * var_one * 2
    emp = float(np.mean(np.square(sq))) if sq else float("nan")
    # Larger secrets make q-wraps visible. The specification's sufficient
    # condition is: every group of four coefficients has at most one wrap.
    stress_fail = stress_ok = condition_violated = 0
    for _ in range(trials):
        f = rng.integers(-8, 9, size=n)
        g = rng.integers(-8, 9, size=n)
        finv_q = _poly_inv_fix(f % q, q)
        f_half = np.array([(int(f[i]) + int(f[i + half])) % 2 for i in range(half)], dtype=np.int64)
        finv = _inv_mod2_gaussian(f_half)
        if finv_q is None or finv is None:
            continue
        s = rng.integers(-8, 9, size=n)
        e = rng.integers(-8, 9, size=n)
        m = rng.integers(0, 2, size=n // 4)
        enc = np.zeros(n, dtype=np.int64)
        base = np.zeros(n, dtype=np.int64)
        for i, bit in enumerate(m):
            if not bit:
                continue
            for t in range(4):
                enc[(i + t * step) % n] = 1
            base[i] = 1
            base[i + step] = 1
        c = (_negacyclic_mul(h if False else _negacyclic_mul(g % q, finv_q, q), s % q, q) + (e % q) + scale * enc) % q
        fc = _negacyclic_mul(f % q, c, q)
        a = np.zeros(n, dtype=np.int64)
        a[:half] = (fc[:half] - fc[half:]) % q
        a[half:] = (fc[:half] + fc[half:]) % q
        a_half = np.array([(_cent(int(a[i]), q) + _cent(int(a[i + half]), q)) % 2 for i in range(half)])
        got = _simple_decoding(a, _mul_mod2_half(a_half, finv), finv, q)
        shifted = np.zeros(n, dtype=np.int64)
        shifted[:half] = -base[half:]
        shifted[half:] = base[:half]
        z = _mul_z(f, shifted) + _mul_z(np.array([1] + [0] * (n - 1)), _mul_z(g, s) + _mul_z(f, e))
        # (1+x^{n/2}) * err
        err = _mul_z(g, s) + _mul_z(f, e)
        wrapped = np.zeros(n, dtype=np.int64)
        wrapped[:half] = err[:half] - err[half:]
        wrapped[half:] = err[:half] + err[half:]
        z = _mul_z(f, shifted) + wrapped
        ewrap = np.array([int(np.round(int(v) / q)) for v in z])
        safe = True
        for r in range(step):
            bits = [int(ewrap[(r + t * step) % n]) != 0 for t in range(4)]
            if sum(bits) >= 2:
                safe = False
        success = np.array_equal(got, m)
        if success:
            stress_ok += 1
        else:
            stress_fail += 1
        if safe and not success:
            condition_violated += 1
    return {
        "scheme": "ZEN-toy",
        "params": {"n": n, "q": q, "secret": "U{-1,0,1}", "compression": "none"},
        "trials": trials,
        "invertible": trials - inv_fail,
        "decrypt_ok": ok,
        "decrypt_fail": fail,
        "empirical_coeff_var": emp,
        "model_coeff_var": model,
        "second_moment_ratio": emp / model if sq else None,
        "stress_secret": "U{-8,...,8}",
        "stress_ok": stress_ok,
        "stress_fail": stress_fail,
        "sufficient_condition_violations": condition_violated,
    }


def run_toys():
    return [toy_scabbard(), toy_rudraksh(), toy_nev(), toy_zen()]


def main():
    import sys
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"
    data = {}
    if mode in ("all", "screen"):
        data.update(run_screen())
    if mode in ("all", "toy"):
        data["toys"] = run_toys()
    if mode == "screen" and os.path.exists(OUT):
        old = json.load(open(OUT))
        old.update(data)
        data = old
    if mode == "toy" and os.path.exists(OUT):
        old = json.load(open(OUT))
        old.update(data)
        data = old
    json.dump(data, open(OUT, "w"), indent=2)
    _print(data)


def _print(data):
    if "minal" in data:
        print(f"{'set':22} {'claim':7} {'model':7} {'w':6} {'p(max)':8} {'64 pre/cls':16} {'80 pre/cls':16}")
        for r in data["minal"]:
            b64, b80 = r["budgets"]["64"], r["budgets"]["80"]
            def fmt(b):
                if not b["reachable"]:
                    return "unreachable"
                return f"{b['precomp_bits']:.0f}/{b['classical_bits']:.0f}"
            m80 = r["model_budgets"]["80"]
            print(f"{r['name']:22} {r['claimed_log2_dfr']:7} {r['model_log2_dfr']:7.1f} {r['variance_fraction_controlled']:6.2f} "
                  f"{r['log2_p_at_max_secret']:8.1f} {fmt(b64):16} {fmt(b80):16} {fmt(m80):16}")
    if "nev" in data:
        print("\nNEV")
        for r in data["nev"]:
            print(f"{r['name']:8} mu {r['mu_sampled']:.2f} (pub {r['mu_published']}) "
                  f"64: pre {r['64']['precomp_bits']:.0f} cls {r['64']['classical_bits']:.0f}  "
                  f"80: pre {r['80']['precomp_bits']:.0f} cls {r['80']['classical_bits']:.0f} (pub {r['published_classical']})")
    if "zen" in data:
        print("\nZEN table-2 arithmetic")
        for r in data["zen"]:
            print(f"{r['name']:28} check {r['classical_check']:.1f}/{r['quantum_check']:.1f} pub {r['published_classical']}/{r['published_quantum']}")
    if "toys" in data:
        print("\ntoys")
        for t in data["toys"]:
            print(t["scheme"], {k: t[k] for k in t if k != "params"})


if __name__ == "__main__":
    main()
