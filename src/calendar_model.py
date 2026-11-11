"""Calendar-aging model and estimation utilities for the Stroebl et al. (2024) dataset.

Model (power law in time, Arrhenius in temperature, log-linear in storage SoC):

    Q_rel(t) = 1 - k(T, SoC) * (t^z - t0^z)
    log k    = log_k_ref - (Ea / R) * (1/T - 1/T_ref) + c_soc * (SoC - 0.5)

Q_rel is normalized to each cell's first regular check-up at time t0, so the
model predicts fade accrued after t0. Time is in years.
"""
import numpy as np
import pandas as pd
from scipy.optimize import least_squares

R = 8.314462618          # J / (mol K)
T_REF = 296.15           # K (23 degC)
REGULAR = ["CU", "exCU"]
PARAM_NAMES = ["log_k_ref", "Ea_kJ", "c_soc", "z"]


def load_calendar(path, stage=1, sampling="FF"):
    """Regular check-ups for one calendar arm, with Q_rel, time in years, and condition label."""
    cap = pd.read_parquet(path)
    d = cap[(cap["type"] == "k") & (cap["stage"] == stage) & (cap["sampling"] == sampling)
            & cap["kind"].isin(REGULAR) & (cap["n_dis_rows"] > 0)].copy()
    d = d.sort_values(["cell_id", "days"])
    first = d.groupby("cell_id")[["Q_Ah", "days"]].transform("first")
    d["Q_rel"] = d["Q_Ah"] / first["Q_Ah"]
    d["t"] = d["days"] / 365.25
    d["t0"] = first["days"] / 365.25
    d["T_K"] = d["amb_temp_tp"] + 273.15
    d["soc"] = d["soc_max_tp"]
    d["cond"] = d["amb_temp_tp"].astype(int).astype(str) + "C/" + d["soc"].map("{:.2f}".format)
    d = d[d["t"] > d["t0"]]                      # reference row carries no information
    return d.reset_index(drop=True)


def predict_param(theta, d):
    log_k, ea_kj, c_soc, z = theta
    log_rate = log_k - (ea_kj * 1e3 / R) * (1 / d["T_K"].values - 1 / T_REF) + c_soc * (d["soc"].values - 0.5)
    return 1 - np.exp(log_rate) * (d["t"].values ** z - d["t0"].values ** z)


def predict_saturated(theta, d, conds):
    """One free rate per condition, shared exponent z (condition effects unconstrained)."""
    idx = pd.Categorical(d["cond"], categories=conds).codes
    log_k, z = theta[:-1], theta[-1]
    return 1 - np.exp(log_k[idx]) * (d["t"].values ** z - d["t0"].values ** z)


def fit(predict, d, x0, **kw):
    """Nonlinear least squares; returns estimate, Fisher (Gauss-Newton) covariance, RSS, n, p."""
    res = least_squares(lambda th: predict(th, d, **kw) - d["Q_rel"].values, x0, method="lm")
    n, p = len(d), len(x0)
    rss = float(res.fun @ res.fun)
    s2 = rss / (n - p)
    cov = s2 * np.linalg.inv(res.jac.T @ res.jac)
    return {"theta": res.x, "cov": cov, "rss": rss, "n": n, "p": p, "resid": res.fun}


def cluster_bootstrap(predict, d, theta_hat, B=500, seed=0, **kw):
    """Resample cells with replacement within each condition and refit; returns B x p draws."""
    rng = np.random.default_rng(seed)
    cells = d.groupby("cond")["cell_id"].unique()
    by_cell = {c: g for c, g in d.groupby("cell_id")}
    draws = []
    for _ in range(B):
        parts = []
        for j, ids in enumerate(cells):
            for k, c in enumerate(rng.choice(ids, size=len(ids), replace=True)):
                parts.append(by_cell[c].assign(cell_id=f"{c}#{j}_{k}"))
        db = pd.concat(parts, ignore_index=True)
        res = least_squares(lambda th: predict(th, db, **kw) - db["Q_rel"].values, theta_hat, method="lm")
        draws.append(res.x)
    return np.array(draws)


SOC_REF = 0.5


def soc_levels(d):
    """Non-reference SoC levels, sorted; SoC 0.5 is the baseline with zero effect."""
    return [s for s in sorted(d["soc"].round(2).unique()) if s != SOC_REF]


def soc_param_names(levels):
    return ["log_k_ref", "Ea_kJ"] + [f"delta_soc{s:.2f}" for s in levels] + ["z"]


def predict_soc_levels(theta, d, levels):
    """Arrhenius in temperature, free additive effect on log k for each SoC level (additive in log k)."""
    log_k, ea_kj, z = theta[0], theta[1], theta[-1]
    delta = np.zeros(len(d))
    soc = d["soc"].values.round(2)
    for s, dl in zip(levels, theta[2:-1]):
        delta[soc == s] = dl
    log_rate = log_k - (ea_kj * 1e3 / R) * (1 / d["T_K"].values - 1 / T_REF) + delta
    return 1 - np.exp(log_rate) * (d["t"].values ** z - d["t0"].values ** z)


def _marginal_nll(par, predict, d, codes, n_cells, gh, kw):
    """Negative marginal log-likelihood; cell effect b ~ N(0, tau^2) on log k, integrated by adaptive Gauss-Hermite.

    The data are far more informative about b than the prior (likelihood width ~ sigma / (fade sqrt(n)) << tau), so
    nodes are centred at each cell's posterior mode and scaled by its curvature (a Laplace-centred rule).
    """
    theta, sigma, tau = par[:-2], np.exp(par[-2]), np.exp(par[-1])
    fade = 1 - predict(theta, d, **kw)
    y_fade = 1 - d["Q_rel"].values
    x, w = gh

    def derivs(b):
        # per-cell log-posterior gradient and curvature in b; fade scales by exp(b) since log k shifts by b
        u = fade * np.exp(b[codes])
        r = y_fade - u
        g = np.bincount(codes, r * u, n_cells) / sigma**2 - b / tau**2
        h = np.bincount(codes, r * u - u**2, n_cells) / sigma**2 - 1 / tau**2
        return g, h

    b = np.zeros(n_cells)
    for _ in range(50):                                          # Newton for the per-cell posterior mode
        g, h = derivs(b)
        step = np.clip(g / h, -0.5, 0.5)
        b = b - step
        if np.abs(step).max() < 1e-10:
            break
    _, h = derivs(b)
    s = 1 / np.sqrt(-h)
    nodes = b[:, None] + np.sqrt(2) * s[:, None] * x[None, :]     # n_cells x n_nodes
    u = fade[:, None] * np.exp(nodes[codes])                       # n_obs x n_nodes
    ll_obs = -0.5 * ((y_fade[:, None] - u) / sigma) ** 2 - np.log(sigma) - 0.5 * np.log(2 * np.pi)
    ll_cell = np.zeros((n_cells, len(x)))
    np.add.at(ll_cell, codes, ll_obs)
    log_prior = -0.5 * (nodes / tau) ** 2 - np.log(tau) - 0.5 * np.log(2 * np.pi)
    # int f(b) db ~ sum_i w_i exp(x_i^2) f(m + sqrt2 s x_i) sqrt2 s
    a = ll_cell + log_prior + np.log(w)[None, :] + x[None, :] ** 2 + np.log(np.sqrt(2) * s)[:, None]
    m = a.max(axis=1, keepdims=True)
    return -float((m[:, 0] + np.log(np.exp(a - m).sum(axis=1))).sum())


def _num_hessian(f, x, h=1e-4):
    p = len(x)
    H = np.zeros((p, p))
    step = h * np.maximum(1.0, np.abs(x))
    for i in range(p):
        for j in range(i, p):
            ei, ej = np.eye(p)[i] * step[i], np.eye(p)[j] * step[j]
            H[i, j] = H[j, i] = (f(x + ei + ej) - f(x + ei - ej) - f(x - ei + ej) + f(x - ei - ej)) / (4 * step[i] * step[j])
    return H


def fit_mixed(predict, d, theta0, tau0=0.1, n_gh=15, **kw):
    """Nonlinear mixed model with a cell-level random effect on log k (Gaussian noise, ML).

    Returns fixed-effect estimates with inverse-Hessian SEs, sigma, tau, and the maximized log-likelihood.
    """
    from scipy.optimize import minimize
    codes = pd.Categorical(d["cell_id"]).codes
    n_cells = codes.max() + 1
    gh = np.polynomial.hermite.hermgauss(n_gh)
    f = lambda par: _marginal_nll(par, predict, d, codes, n_cells, gh, kw)
    base = fit(predict, d, theta0, **kw)
    s0 = np.sqrt(base["rss"] / (base["n"] - base["p"]))
    x0 = np.r_[base["theta"], np.log(s0), np.log(tau0)]
    res = minimize(f, x0, method="BFGS", options={"gtol": 1e-6})
    res = minimize(f, res.x, method="Nelder-Mead", options={"xatol": 1e-7, "fatol": 1e-10, "maxiter": 20000})
    cov = np.linalg.inv(_num_hessian(f, res.x))
    p = len(theta0)
    return {"theta": res.x[:p], "se": np.sqrt(np.diag(cov))[:p], "sigma": np.exp(res.x[-2]), "tau": np.exp(res.x[-1]),
            "loglik": -res.fun, "par": res.x, "cov": cov}


def soc_int_param_names(levels):
    return soc_param_names(levels)[:-1] + ["gamma_Ea_kJ", "z"]


def predict_soc_int(theta, d, levels):
    """Per-SoC rate effects plus an activation energy that varies linearly with SoC: Ea = Ea0 + gamma (SoC - 0.5)."""
    log_k, ea_kj, gamma, z = theta[0], theta[1], theta[-2], theta[-1]
    soc = d["soc"].values.round(2)
    delta = np.zeros(len(d))
    for s, dl in zip(levels, theta[2:-2]):
        delta[soc == s] = dl
    ea = (ea_kj + gamma * (soc - SOC_REF)) * 1e3
    log_rate = log_k - (ea / R) * (1 / d["T_K"].values - 1 / T_REF) + delta
    return 1 - np.exp(log_rate) * (d["t"].values ** z - d["t0"].values ** z)
