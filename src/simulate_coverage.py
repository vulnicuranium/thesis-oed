"""Coverage simulation for the calendar-aging model (C2).

Truth is the cell-level mixed model fitted to the Stage-1 FF data (per-SoC-level effects). Each replicate redraws
cell effects b_c ~ N(0, tau^2) and noise ~ N(0, sigma^2) on the observed design, then refits and builds 95%
intervals from Fisher (NLS Gauss-Newton), cell-cluster bootstrap (percentile and Wald), and the mixed-model Hessian.
"""
import sys
import numpy as np
import pandas as pd
from multiprocessing import Pool
from calendar_model import (load_calendar, soc_levels, soc_param_names, predict_soc_levels, fit, fit_mixed,
                            cluster_bootstrap)

PARQUET = "../data/processed/capacity_rpt.parquet"
OUT = "../data/processed/sim_coverage_soc.parquet"
D = load_calendar(PARQUET, stage=1, sampling="FF")
LV = soc_levels(D)
NAMES = soc_param_names(LV)
X0 = np.r_[np.log(0.02), 30.0, np.zeros(len(LV)), 0.6]


def truth():
    f = fit(predict_soc_levels, D, X0, levels=LV)
    m = fit_mixed(predict_soc_levels, D, f["theta"], levels=LV)
    return m["theta"], m["sigma"], m["tau"]


def simulate(theta, sigma, tau, rng):
    d = D.copy()
    b = pd.Series(rng.normal(0, tau, d["cell_id"].nunique()), index=d["cell_id"].unique())
    fade = 1 - predict_soc_levels(theta, d, LV)
    d["Q_rel"] = 1 - fade * np.exp(d["cell_id"].map(b).values) + rng.normal(0, sigma, len(d))
    return d


def one_rep(args):
    i, theta, sigma, tau, B = args
    rng = np.random.default_rng(1000 + i)
    d = simulate(theta, sigma, tau, rng)
    try:
        fl = fit(predict_soc_levels, d, theta, levels=LV)
        mm = fit_mixed(predict_soc_levels, d, fl["theta"], levels=LV)
        draws = cluster_bootstrap(predict_soc_levels, d, fl["theta"], B=B, seed=i, levels=LV)
    except Exception as e:                                       # a failed replicate is recorded, not raised
        print(f"rep {i} failed: {e}", file=sys.stderr)
        return None
    se_f, se_b, se_m = np.sqrt(np.diag(fl["cov"])), draws.std(axis=0, ddof=1), mm["se"]
    plo, phi = np.percentile(draws, [2.5, 97.5], axis=0)
    rows = []
    for j, n in enumerate(NAMES):
        rows.append(dict(rep=i, param=n, truth=theta[j], est_nls=fl["theta"][j], est_mixed=mm["theta"][j],
                         se_fisher=se_f[j], se_boot=se_b[j], se_mixed=se_m[j], pct_lo=plo[j], pct_hi=phi[j],
                         tau_hat=mm["tau"], sigma_hat=mm["sigma"]))
    return rows


if __name__ == "__main__":
    R, B = int(sys.argv[1]), int(sys.argv[2])
    theta, sigma, tau = truth()
    print(f"truth: sigma={sigma:.5f} tau={tau:.4f}", flush=True)
    with Pool() as p:
        res = p.map(one_rep, [(i, theta, sigma, tau, B) for i in range(R)])
    rows = [r for rr in res if rr for r in rr]
    pd.DataFrame(rows).to_parquet(OUT)
    print(f"saved {len(rows) // len(NAMES)} / {R} replicates to {OUT}")
