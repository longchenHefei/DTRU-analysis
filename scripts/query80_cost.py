"""Offline cost of a ciphertext with failure probability 2^{-80}.

The score is the ciphertext coefficient squares. alpha is the probability of
reaching the squared-norm that makes the decoder fail with probability 2^{-80}.
Queries are capped at 2^80. If no legal ciphertext reaches that probability,
the row records the failure probability at the maximum legal squares.
"""

import json
import math
from pathlib import Path

import numpy as np

OUT = Path(__file__).with_name("query80_cost.json")
CAP = 80


def logsumexp(xs):
    m = max(xs)
    if m == -math.inf:
        return -math.inf
    return m + math.log(sum(math.exp(x - m) for x in xs))


def log_erfc(x):
    """log(erfc(x)) for x >= 0. erfc(x) = 2 Phi_c(x*sqrt(2)) wait: erfc(x)=P(|N(0,1/2)|>x)*... 
    math.erfc(x) = (2/sqrt(pi)) int_x^inf e^{-t^2} dt = 2 Phi(-x*sqrt(2))? 
    Standard: Phi_c(z) = P(N(0,1) > z) = 0.5*erfc(z/sqrt(2)).
    Here log(erfc(x)) with x = z/sqrt(2), so log P(N>z) = log(0.5*erfc(z/sqrt(2))).
    """
    if x < 0:
        # erfc negative arg = 2 - erfc(|x|), not used for tails
        return math.log(math.erfc(x))
    if x < 5:
        return math.log(math.erfc(x))
    # erfc(x) ~ exp(-x^2)/(x sqrt(pi)) * (1 - 1/(2x^2) + 3/(4x^4))
    corr = 1 - 1 / (2 * x * x) + 3 / (4 * x ** 4)
    return -x * x - math.log(x) - 0.5 * math.log(math.pi) + math.log(corr)


def log_gauss_tail(z):
    """log P(|N(0,1)| > z) = log(erfc(z/sqrt(2)))."""
    if z <= 0:
        return 0.0
    return log_erfc(z / math.sqrt(2))


def log2_coeff_union(log_p_one, npos):
    """log2(1 - (1-p)^n) ≈ log2(n p) when p is tiny."""
    log2_p = log_p_one / math.log(2)
    log2_n = math.log2(npos)
    if log2_p + log2_n < -1:
        return log2_p + log2_n
    p = math.exp(log_p_one)
    return math.log2(1 - (1 - p) ** npos)


# ---------------------------------------------------------------- CBD / ternary squares


def square_atoms_cbd(eta):
    counts = {}
    total = 1 << (2 * eta)
    for mask in range(total):
        s = 0
        for i in range(eta):
            s += (mask >> i) & 1
            s -= (mask >> (eta + i)) & 1
        counts[s * s] = counts.get(s * s, 0) + 1
    return {k: v / total for k, v in counts.items()}


def square_atoms_ternary(sigma):
    # P(±1)=sigma, P(0)=1-2sigma, so P(s^2=1)=2sigma
    return {0: 1 - 2 * sigma, 1: 2 * sigma}


def tilt_cost(atoms, mean_sq, n_coeffs):
    """log2 of the probability that the mean square reaches mean_sq.

    Returns (log2_prob, ). Negative. Includes the local-CLT prefactor.
    """
    keys = sorted(atoms)
    honest = sum(k * atoms[k] for k in keys)
    hi = keys[-1]
    if mean_sq <= honest + 1e-15:
        return 0.0
    if mean_sq >= hi - 1e-12:
        # point mass at the maximum: only the top atom
        p = atoms[hi]
        return n_coeffs * math.log2(p)
    # small deviations: plain normal tail of the sample mean (the saddlepoint
    # prefactor 1/(theta sigma sqrt(2 pi n)) is not valid when theta -> 0)
    var_h = sum(k * k * atoms[k] for k in keys) - honest * honest
    z = (mean_sq - honest) * math.sqrt(n_coeffs) / math.sqrt(max(var_h, 1e-18))
    if z < 3.0:
        return min(0.0, log_gauss_tail_one_sided(z) / math.log(2))
    # theta such that tilted mean equals mean_sq
    def log_mgf_parts(theta):
        logs = [math.log(atoms[k]) + theta * k for k in keys]
        lmgf = logsumexp(logs)
        # tilted probabilities
        ws = [lp - lmgf for lp in logs]
        mean = sum(k * math.exp(w) for k, w in zip(keys, ws))
        second = sum(k * k * math.exp(w) for k, w in zip(keys, ws))
        return lmgf, mean, second

    lo, hi_t = 0.0, 1.0
    for _ in range(60):
        _, mean, _ = log_mgf_parts(hi_t)
        if mean >= mean_sq:
            break
        hi_t *= 2.0
        if hi_t > 40:
            break
    for _ in range(70):
        mid = 0.5 * (lo + hi_t)
        _, mean, _ = log_mgf_parts(mid)
        if mean < mean_sq:
            lo = mid
        else:
            hi_t = mid
    theta = 0.5 * (lo + hi_t)
    lmgf, mean, second = log_mgf_parts(theta)
    var = max(second - mean * mean, 1e-18)
    rate = theta * mean_sq - lmgf  # nats per coefficient
    # P(mean >= a) ~ exp(-n rate) / (theta * sigma * sqrt(2 pi n))
    pref = math.log(theta * math.sqrt(var) * math.sqrt(2 * math.pi * n_coeffs))
    return min(0.0, -(n_coeffs * rate + pref) / math.log(2))


# ---------------------------------------------------------------- tails


def log_abs_sum_tail(sigma, repeats, threshold):
    """log P(sum of `repeats` independent |N(0, sigma^2)| >= threshold)."""
    if threshold <= 0:
        return 0.0

    def log_mgf(theta):
        # log E exp(theta |Z|)
        a = sigma * theta / math.sqrt(2)
        # exp(sigma^2 theta^2 / 2) * (1+erf(a))
        lg = 0.5 * (sigma * theta) ** 2
        if a > 6:
            return lg + math.log(2.0)
        return lg + math.log(1.0 + math.erf(a))

    def d1(theta):
        # derivative of log mgf of one |Z|, by central difference in a stable way
        h = 1e-4 * (1 + theta)
        return (log_mgf(theta + h) - log_mgf(theta - h)) / (2 * h)

    # find theta with R * mean_|Z| = threshold
    # mean grows like sigma^2 * theta for large theta
    lo, hi = 1e-8, max(10.0, threshold / (repeats * sigma * sigma) * 4 + 1)
    for _ in range(60):
        if d1(hi) * repeats >= threshold:
            break
        hi *= 2
        if hi > 1e4:
            break
    for _ in range(70):
        mid = 0.5 * (lo + hi)
        if d1(mid) * repeats < threshold:
            lo = mid
        else:
            hi = mid
    theta = 0.5 * (lo + hi)
    # second derivative
    h = 1e-3 * (1 + theta)
    second = (log_mgf(theta + h) - 2 * log_mgf(theta) + log_mgf(theta - h)) / (h * h)
    second = max(second, 1e-18)
    kappa = repeats * log_mgf(theta)
    rate = theta * threshold - kappa  # positive nats for the upper tail
    pref = math.log(theta * math.sqrt(2 * math.pi * repeats * second))
    return -(rate + pref)


def log_gauss_plus_uniform(var_g, half_width, threshold):
    """log P(|G+U| >= threshold), G~N(0,var_g), U uniform on [-half_width, half_width]."""
    sigma = math.sqrt(var_g)
    # quadrature on [0, B]; density even
    b = half_width
    if b <= 0:
        return log_gauss_tail(threshold / sigma)
    nodes = 96
    xs = [(i + 0.5) / nodes * b for i in range(nodes)]
    logs = []
    for u in xs:
        # even in u: average of P(|G+u|>=T) over [0,B] equals the average over [-B,B]
        z1 = (threshold - u) / sigma
        z2 = (threshold + u) / sigma
        log_p = logsumexp([
            log_gauss_tail_one_sided(z1),
            log_gauss_tail_one_sided(z2),
        ])
        logs.append(log_p)
    # average. each node has width b/nodes, total measure from 0 to B is half the mass,
    # and we used the even reduction: average over [-B,B] equals average over [0,B]
    # of the same even function. So just mean of log-sum-exp.
    return logsumexp([lp - math.log(nodes) for lp in logs])


def log_gauss_tail_one_sided(z):
    """log P(N(0,1) > z)."""
    if z < -8:
        return 0.0
    if z < 0:
        # 1 - Phi_c(|z|) = Phi(z) wait P(N>z) for z<0 is > 1/2
        return math.log(0.5 * math.erfc(z / math.sqrt(2)))
    return log_erfc(z / math.sqrt(2)) - math.log(2)


def log2_chi2_sf(x, k):
    """log2 P(chi^2_k >= x) for even k, via the Poisson sum."""
    if x <= k:
        return 0.0
    half = k / 2.0
    m = int(half) - 1
    lt = -x / 2.0
    terms = [lt]
    for j in range(1, m + 1):
        lt = lt + math.log(x / 2.0) - math.log(j)
        terms.append(lt)
    return logsumexp(terms) / math.log(2)


def bw_block_log2(var_g, half_u, n_dim, radius):
    """log2 P(||G+U|| >= radius). U_i uniform[-B,B] independent, or chi^2 if B is small.

    Uses the chi-square formula on var_g+var(U) when the uniform piece is under
    15% of the variance (c256, c512). Otherwise a saddlepoint on the squared
    norm, averaging the noncentral chi-square mgf over the uniform.
    """
    var_u = (2 * half_u) ** 2 / 12.0 if half_u > 0 else 0.0
    total = var_g + var_u
    if half_u <= 0 or var_u < 0.15 * total:
        x = radius ** 2 / total
        return log2_chi2_sf(x, n_dim)
    # mgf of one squared coordinate, theta < 1/(2 var_g)
    v = var_g
    b = half_u
    theta_max = 0.45 / v

    def log_mgf_one(theta):
        # average over u in [-B,B] of log-space mixture
        # E[e^{theta (g+u)^2}] = exp(theta u^2 / (1-2 theta v)) / sqrt(1-2 theta v)
        nodes = 64
        logs = []
        denom = 1 - 2 * theta * v
        log_norm = -0.5 * math.log(denom)
        for i in range(nodes):
            u = -b + (i + 0.5) * (2 * b) / nodes
            logs.append(theta * u * u / denom + log_norm)
        return logsumexp([lp - math.log(nodes) for lp in logs])

    def d1(theta):
        h = theta_max * 1e-4
        return (log_mgf_one(theta + h) - log_mgf_one(theta - h)) / (2 * h)

    target = radius ** 2 / n_dim  # mean of one square
    lo, hi = 1e-10, theta_max * 0.98
    if d1(hi) < target:
        # cannot reach; return a very small number via the chi2 bound shifted by the max uniform
        # fall back to gaussian with the truncated second moment
        return log2_chi2_sf(radius ** 2 / total, n_dim)
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        if d1(mid) < target:
            lo = mid
        else:
            hi = mid
    theta = 0.5 * (lo + hi)
    h = theta_max * 1e-3
    second = (log_mgf_one(theta + h) - 2 * log_mgf_one(theta) + log_mgf_one(theta - h)) / (h * h)
    second = max(second, 1e-18)
    kappa = n_dim * log_mgf_one(theta)
    rate = theta * (radius ** 2) - kappa
    pref = math.log(theta * math.sqrt(2 * math.pi * n_dim * second))
    return -(rate + pref) / math.log(2)


# ---------------------------------------------------------------- schemes


def cheapest(parts, target_var, budget):
    """parts: list of (n_coeffs, atoms, var_at_S1).

    var = fixed + sum var_at_S1 * S, S = mean_sq / honest_mean.
    Return log2 alpha (negative or zero) of the cheapest way to reach target_var.
    None if unreachable.
    """
    from scipy.optimize import minimize

    fixed = budget
    meta = []
    for n, atoms, v in parts:
        honest = sum(k * p for k, p in atoms.items())
        hi = max(atoms)
        meta.append((n, atoms, v, honest, hi / honest))
    # variance at the legal maximum
    vmax = fixed + sum(v * smax for _, _, v, _, smax in meta)
    if vmax < target_var * (1 - 1e-9):
        return None
    if fixed >= target_var:
        return (0.0, tuple(1.0 for _ in meta), fixed)

    def objective(x):
        # x in R^d, map through a sigmoid into [1, smax]
        cost = 0.0
        var = fixed
        for (n, atoms, v, honest, smax), t in zip(meta, x):
            S = 1.0 + (smax - 1.0) * (1 / (1 + math.exp(-t)))
            var += v * S
            cost += tilt_cost(atoms, min(S, smax * (1 - 1e-9)) * honest, n)
        # cost is log2 alpha <= 0. Penalise a variance shortfall heavily.
        short = max(0.0, target_var - var)
        return -cost + (short / max(target_var, 1.0)) * 1e6

    x0 = np.zeros(len(meta))
    res = minimize(objective, x0, method="Nelder-Mead")
    # evaluate the feasible point
    Ss = []
    var = fixed
    cost = 0.0
    for (n, atoms, v, honest, smax), t in zip(meta, res.x):
        S = 1.0 + (smax - 1.0) * (1 / (1 + math.exp(-float(t))))
        # if the optimiser stopped short, push the most efficient coordinate
        Ss.append(S)
        var += v * S
        cost += tilt_cost(atoms, min(S, smax * (1 - 1e-9)) * honest, n)
    if var < target_var:
        # fall back: scale every S up uniformly until the variance is met
        lo, hi = 0.0, 1.0
        for _ in range(40):
            mid = 0.5 * (lo + hi)
            var_m = fixed + sum(v * (1 + (smax - 1) * mid) for _, _, v, _, smax in meta)
            if var_m < target_var:
                lo = mid
            else:
                hi = mid
        cost = 0.0
        Ss = []
        for n, atoms, v, honest, smax in meta:
            S = 1 + (smax - 1) * hi
            Ss.append(S)
            cost += tilt_cost(atoms, min(S, smax * (1 - 1e-9)) * honest, n)
    return (cost, tuple(Ss), var)


def solve_target(log2_of_var, lo, hi, goal=-CAP):
    """Binary search var in [lo, hi] so log2_of_var(var) ~= goal. log2_of_var increases with var."""
    f_lo = log2_of_var(lo)
    f_hi = log2_of_var(hi)
    if f_hi < goal:
        return None, f_hi
    if f_lo >= goal:
        return lo, f_lo
    a, b = lo, hi
    for _ in range(50):
        m = 0.5 * (a + b)
        if log2_of_var(m) < goal:
            a = m
        else:
            b = m
    return b, log2_of_var(b)


def row(name, level, claimed_delta, claimed_c, claimed_q, core_c, core_q,
        model_log2, max_log2, precomp, note):
    # A negative model_log2 is log2 of the honest failure probability.
    # If that is already at least 2^{-80}, random ciphertexts meet the cap.
    if model_log2 >= -CAP:
        precomp = 0.0
        queries = -model_log2
        total = queries
        quantum = queries
        reachable = True
    elif max_log2 >= -CAP - 0.05 and precomp is not None:
        queries = CAP
        total = -precomp + CAP
        quantum = -0.5 * precomp + CAP
        reachable = True
    else:
        queries = None
        total = None
        quantum = None
        reachable = False
    return {
        "name": name,
        "level": level,
        "claimed_log2_delta": claimed_delta,
        "model_log2_delta": round(model_log2, 2),
        "max_ciphertext_log2": round(max_log2, 2),
        "reachable_2_80": bool(reachable and precomp is not None),
        "log2_alpha": None if precomp is None else round(precomp, 2),
        "queries": None if queries is None else round(queries, 2),
        "classical": None if total is None else round(total, 2),
        "quantum": None if quantum is None else round(quantum, 2),
        "claimed_C": claimed_c,
        "claimed_Q": claimed_q,
        "core_svp_C": core_c,
        "core_svp_Q": core_q,
        "below_nominal": bool(total is not None and total < level),
        "below_core_svp": bool(total is not None and total < core_c),
        "note": note,
    }


# ---------------------------------------------------------------- FLIT


def flit_variance(N, tf, tg, tr, te, q, d, message_sq=0.5):
    ef, eg, er, ee = 2 * tf, 2 * tg, 2 * tr, 2 * te
    var_gr = N * eg * er
    var_fe = N * ef * ee
    var_ep = (q / 2 ** d) ** 2 / 12.0
    var_fep = N * ef * var_ep
    var_floor = N * ef * message_sq / 4.0
    return {
        "gr": var_gr,
        "fe": var_fe,
        "fep": var_fep,
        "floor": var_floor,
        "total": var_gr + var_fe + var_fep + var_floor,
    }


def screen_flit():
    sets = [
        # name, N, n, q, tf, tg, tr, te, d, claimed, Cc, Cq, coreC, coreQ, level
        ("Flit128", 512, 256, 769, 1/8, 5/16, 1/8, 5/16, 8, -187.5, 118.8, 104.6, 120.6, 109.5, 128),
        ("Flit256", 1024, 256, 769, 5/32, 1/4, 5/32, 1/4, 8, -176.1, 267.2, 235.2, 329.1, 298.7, 256),
        ("Flit512", 2048, 512, 3329, 7/16, 7/16, 7/16, 7/16, 9, -195.5, 550.4, 484.4, 511.9, 464.6, 512),
    ]
    out = []
    for name, N, n, q, tf, tg, tr, te, d, claimed, Cc, Cq, coreC, coreQ, level in sets:
        R = N // n
        threshold = R * (q - 1) / 4.0
        base = flit_variance(N, tf, tg, tr, te, q, d, 0.5)
        # attacker can set every message bit, doubling the floor term, and push r, e to ±1
        top = flit_variance(N, tf, tg, tr, te, q, d, 1.0)
        smax_r = 1.0 / (2 * tr)
        smax_e = 1.0 / (2 * te)
        var_max = top["fep"] + top["floor"] + top["gr"] * smax_r + top["fe"] * smax_e
        # honest uses message 1/2; the floor term does not scale with r, e
        fixed = base["fep"] + base["floor"]

        def log2_fail(var):
            return log_abs_sum_tail(math.sqrt(var), R, threshold) / math.log(2) + math.log2(n)

        model = log2_fail(base["total"])
        at_max = log2_fail(var_max)
        precomp = None
        if at_max >= -CAP:
            target, _ = solve_target(log2_fail, base["total"], var_max, -CAP)
            # var = fixed + gr*S_r + fe*S_e
            parts = []
            if base["gr"] > 0:
                parts.append((N, square_atoms_ternary(tr), base["gr"]))
            if base["fe"] > 0:
                parts.append((N, square_atoms_ternary(te), base["fe"]))
            best = cheapest(parts, target, fixed)
            precomp = best[0] if best else None
        out.append(row(name, level, claimed, Cc, Cq, coreC, coreQ, model, at_max, precomp,
                        "L1 of repetition block, product coefficients uncorrelated"))
    return out


# ---------------------------------------------------------------- Cheetah


def screen_cheetah():
    q = 7681
    dq = math.ceil(math.log2(q))
    N = 640
    sets = [
        # k, lam, eta, db, du, dv, claimed, Cc, Cq, coreC, coreQ, level
        ("Cheetah128", 1, 128, 5, 10, 10, 4, -129, 159, 139, 148.3, 134.6, 128),
        ("Cheetah256", 2, 256, 4, 10, 10, 4, -176, 307, 276, 325.9, 295.7, 256),
        ("Cheetah384", 3, 384, 3, 11, 11, 4, -189, 426, 382, 499.0, 452.9, 384),
        ("Cheetah512", 4, 512, 2, 11, 11, 8, -243, 541, 497, 511.9, 464.6, 512),
    ]
    out = []
    threshold = q / 4.0
    for name, k, lam, eta, db, du, dv, claimed, Cc, Cq, coreC, coreQ, level in sets:
        sig2 = eta / 2.0
        Db = 2 ** (dq - db)
        Du = 2 ** (dq - du)
        Dv = 2 ** (dq - dv)
        var_eb = Db ** 2 / 12.0
        var_eu = Du ** 2 / 12.0
        half_v = Dv / 2.0  # |e'| <= Dv/2
        # gaussian pieces at S=1
        var_er = k * N * sig2 * sig2          # e·r
        var_ebr = k * N * var_eb * sig2       # e_b·r
        var_se1 = k * N * sig2 * sig2         # s·e1
        var_seu = k * N * sig2 * var_eu       # s·e''
        var_e2 = sig2
        var_g = var_er + var_ebr + var_se1 + var_seu + var_e2
        npos = lam

        def log2_fail_from_g(vg):
            lp = log_gauss_plus_uniform(vg, half_v, threshold)
            return log2_coeff_union(lp, npos)

        model = log2_fail_from_g(var_g)
        # max: r and e1 at Smax, e2 ignored (one polynomial, variance sig2)
        smax = (eta * eta) / sig2
        var_max = var_seu + var_e2 + (var_er + var_ebr) * smax + var_se1 * smax
        at_max = log2_fail_from_g(var_max)
        precomp = None
        if at_max >= -CAP:
            target, _ = solve_target(log2_fail_from_g, var_g, var_max, -CAP)
            parts = [
                (k * N, square_atoms_cbd(eta), var_er + var_ebr),
                (k * N, square_atoms_cbd(eta), var_se1),
            ]
            best = cheapest(parts, target, var_seu + var_e2)
            precomp = best[0] if best else None
        out.append(row(
            name, level, claimed, Cc, Cq, coreC, coreQ, model, at_max, precomp,
            f"threshold q/4, v-compression uniform half-width {half_v:.0f}, other terms gaussian",
        ))
    return out


# ---------------------------------------------------------------- BW-KEM


def screen_bw():
    q = 3329
    qhat = 4096
    tau = 2
    sets = [
        # name, N, n, l, es, ee, ect, du, dv, claimed, Cc, Cq, coreC, coreQ, level
        ("BW-c128", 256, 8, 2, 6, 6, 5, 10, 4, -138.14, 131, 119, 137.8, 125.1, 128),
        ("BW-c256", 256, 32, 4, 3, 3, 3, 10, 5, -209.94, 267, 242, 291.7, 264.7, 256),
        ("BW-c384", 512, 32, 3, 2, 2, 2, 10, 4, -216.26, 403, 365, 445.6, 404.4, 384),
        ("BW-c512", 512, 32, 4, 2, 2, 2, 10, 6, -232.71, 556, 504, 511.9, 464.6, 512),
    ]
    out = []
    for name, N, n, l, es, ee, ect, du, dv, claimed, Cc, Cq, coreC, coreQ, level in sets:
        radius = q * math.sqrt(2 * n) / 2 ** (tau + 2) - math.sqrt(n) / 2 * (q / qhat + 1)
        blocks = N // n
        vr = l * N * (ee / 2) * (ect / 2)
        ve1 = l * N * (es / 2) * (ect / 2)
        veu = l * N * (es / 2) * ((q / 2 ** du) ** 2 / 12)
        ve2 = ect / 2
        half_v = q / 2 ** (dv + 1)
        var_g0 = vr + ve1 + veu + ve2

        def log2_fail(vg):
            return bw_block_log2(vg, half_v, n, radius) + math.log2(blocks)

        model = log2_fail(var_g0)
        smax = (ect * ect) / (ect / 2)
        var_max = veu + ve2 + vr * smax + ve1 * smax
        at_max = log2_fail(var_max)
        precomp = None
        if at_max >= -CAP:
            target, _ = solve_target(log2_fail, var_g0, var_max, -CAP)
            parts = [
                (l * N, square_atoms_cbd(ect), vr),
                (l * N, square_atoms_cbd(ect), ve1),
            ]
            best = cheapest(parts, target, veu + ve2)
            precomp = best[0] if best else None
        note = "packing radius of Lemma 2; block is n real coefficients"
        if name == "BW-c128":
            note += "; mu0=4 of capacity 12, published delta matches a larger radius"
        out.append(row(name, level, claimed, Cc, Cq, coreC, coreQ, model, at_max, precomp, note))
    return out


# ---------------------------------------------------------------- NTRE


def screen_ntre():
    # conservative model of the spec: each coefficient of gr+f'm is a sum of n
    # atoms, var = 0.75 n. Both products scale with the ciphertext CBD1 squares.
    sets = [
        ("NTRE-2593-648", 648, 2593, -549, 132, 118, 149.5, 135.7, 128),
        ("NTRE-2917-1296", 1296, 2917, -370, 290, 259, 333.5, 302.6, 256),
        ("NTRE-3457-2304", 2304, 3457, -298, 584, 513, 512.5, 465.1, 512),
    ]
    out = []
    atoms = square_atoms_cbd(1)
    for name, n, q, claimed, Cc, Cq, coreC, coreQ, level in sets:
        var0 = 0.75 * n
        # two products, equal variance, each scales with its own ciphertext vector
        # S_max = 1 / (1/2) = 2
        threshold = (q - 2) / 4.0

        def log2_fail(var):
            sigma = math.sqrt(var)
            # two-sided gaussian tail, union over 2n coefficients (Lemma 2.1)
            lp = log_gauss_tail(threshold / sigma)
            return log2_coeff_union(lp, 2 * n)

        model = log2_fail(var0)
        at_max = log2_fail(var0 * 2)  # both vectors at S=2
        precomp = None
        if at_max >= -CAP:
            target, _ = solve_target(log2_fail, var0, var0 * 2, -CAP)
            # var = (var0/2)*S_r + (var0/2)*S_m
            parts = [
                (n, atoms, var0 / 2),
                (n, atoms, var0 / 2),
            ]
            best = cheapest(parts, target, 0.0)
            precomp = best[0] if best else None
        out.append(row(
            name, level, claimed, Cc, Cq, coreC, coreQ, model, at_max, precomp,
            "spec convolution of n atoms; both ciphertext CBD1 vectors scale, S_max=2",
        ))
    return out


def main():
    rows = screen_flit() + screen_cheetah() + screen_bw() + screen_ntre()
    OUT.write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n")
    print(f"{'name':22} {'model':>8} {'claimed':>8} {'at_max':>8} {'pre':>8} {'cls':>8} {'below':>6}")
    for r in rows:
        print(f"{r['name']:22} {r['model_log2_delta']:8.1f} {r['claimed_log2_delta']:8.1f} "
              f"{r['max_ciphertext_log2']:8.1f} "
              f"{str(r['log2_alpha']):>8} {str(r['classical']):>8} {str(r['below_core_svp']):>6}")


if __name__ == "__main__":
    main()
