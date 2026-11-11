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
