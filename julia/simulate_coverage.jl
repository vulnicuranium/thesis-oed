# Coverage simulation for the calendar-aging model (C2); Julia port of src/simulate_coverage.py.
#
# Each replicate redraws cell effects b_c ~ N(0, tau^2) and noise ~ N(0, sigma^2) on the observed design (33 cells,
# 11 conditions), then refits the per-SoC-level model and builds 95% intervals from Fisher (NLS Gauss-Newton),
# cell-cluster bootstrap (percentile and Wald) and the mixed-model Hessian.
#
# Scenarios (what generates the data):
#   wellspec  truth = cell-level mixed fit of the per-SoC model itself (model correct, tau ~ 0.2).
#   misspec   truth = cell-level mixed fit of the saturated model (free rate per condition, tau ~ 0.04). The analysis
#             model is still per-SoC, so it is misspecified. The target is the pseudo-true parameter: the least-squares
#             projection of the noise-free saturated curves onto the per-SoC model.
#
# Usage (from the repo root, using all cores):
#   julia -t auto --project=julia julia/simulate_coverage.jl [reps=100] [B=500] [scenario=wellspec] [out]
# Default output: data/processed/sim_coverage_soc.parquet (wellspec) or sim_coverage_soc_misspec.parquet (misspec).
# The output has the same columns for both scenarios; notebooks/05_coverage_sim.ipynb reads them.

using Random, Printf, Statistics, LinearAlgebra, DataFrames, Parquet2
include(joinpath(@__DIR__, "src", "CalendarModel.jl"))
using .CalendarModel

const ROOT = normpath(joinpath(@__DIR__, ".."))
const PARQUET = joinpath(ROOT, "data", "processed", "capacity_rpt.parquet")

reps = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 100
B = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 500
scenario = length(ARGS) >= 3 ? ARGS[3] : "wellspec"
scenario in ("wellspec", "misspec") || error("scenario must be wellspec or misspec, got $scenario")
default_out = scenario == "wellspec" ? "sim_coverage_soc.parquet" : "sim_coverage_soc_misspec.parquet"
out = length(ARGS) >= 4 ? ARGS[4] : joinpath(ROOT, "data", "processed", default_out)

D = load_calendar(PARQUET; stage=1, sampling="FF")
names = param_names(D)
x0 = vcat(log(0.02), 30.0, zeros(length(D.levels)), 0.6)
@printf("design: %d rows, %d cells, %d conditions | scenario: %s | threads: %d\n", length(D.y), length(D.cell_ids),
        length(D.conds), scenario, Threads.nthreads())

# Data-generating model: truth_model(theta_gen, D) gives fade; theta_star is the target of the analysis model
if scenario == "wellspec"
    truth_model = fade
    gen = fit_mixed(D, fit_nls(D, x0).theta)
    theta_star = gen.theta
else
    truth_model = fade_saturated
    xs = vcat(fill(log(0.02), length(D.conds)), 0.6)
    gen = fit_mixed(D, fit_nls(D, xs; model=fade_saturated).theta; model=fade_saturated)
    # Pseudo-truth: project the noise-free saturated mean curve onto the per-SoC model
    D_mean = CalendarModel.with_y(D, 1 .- fade_saturated(gen.theta, D))
    theta_star = fit_nls(D_mean, fit_nls(D, x0).theta).theta
end
@printf("generating model: sigma=%.5f tau=%.4f\n", gen.sigma, gen.tau)

function one_rep(i)
    rng = Xoshiro(1000 + i)
    d = simulate(D, gen.theta, gen.sigma, gen.tau, rng; model=truth_model)
    try
        fl = fit_nls(d, theta_star)
        mm = fit_mixed(d, fl.theta)
        draws = cluster_bootstrap(d, fl.theta, B, rng)
        se_f, se_b = sqrt.(diag(fl.cov)), vec(std(draws; dims=1))
        plo = [quantile(draws[:, j], 0.025) for j in axes(draws, 2)]
        phi = [quantile(draws[:, j], 0.975) for j in axes(draws, 2)]
        return [(rep=i - 1, param=names[j], truth=theta_star[j], est_nls=fl.theta[j], est_mixed=mm.theta[j],
                 se_fisher=se_f[j], se_boot=se_b[j], se_mixed=mm.se[j], pct_lo=plo[j], pct_hi=phi[j],
                 tau_hat=mm.tau, sigma_hat=mm.sigma) for j in eachindex(names)]
    catch e                                      # a failed replicate is recorded, not raised
        @printf(stderr, "rep %d failed: %s\n", i - 1, sprint(showerror, e))
        return nothing
    end
end

results = Vector{Any}(undef, reps)
done = Threads.Atomic{Int}(0)
t_start = time()
Threads.@threads for i in 1:reps
    results[i] = one_rep(i)
    k = Threads.atomic_add!(done, 1) + 1
    (k % 100 == 0 || k == reps) && @printf("  %d / %d replicates (%.0f s)\n", k, reps, time() - t_start)
end

rows = reduce(vcat, [r for r in results if r !== nothing]; init=NamedTuple[])
mkpath(dirname(out))
Parquet2.writefile(out, DataFrame(rows))
@printf("saved %d / %d replicates to %s (%.0f s)\n", length(rows) ÷ length(names), reps, out, time() - t_start)
