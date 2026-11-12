# Coverage simulation for the calendar-aging model (C2); Julia port of src/simulate_coverage.py.
#
# Truth is the cell-level mixed model fitted to the Stage-1 FF data (per-SoC-level effects). Each replicate redraws
# cell effects b_c ~ N(0, tau^2) and noise ~ N(0, sigma^2) on the observed design, then refits and builds 95%
# intervals from Fisher (NLS Gauss-Newton), cell-cluster bootstrap (percentile and Wald) and the mixed-model Hessian.
#
# Usage (from the repo root, using all cores):
#   julia -t auto --project=julia julia/simulate_coverage.jl [reps=100] [B=500] [out=data/processed/sim_coverage_soc.parquet]
#
# The output has the same columns as the Python version, so notebooks/05_coverage_sim.ipynb reads it unchanged.

using Random, Printf, Statistics, LinearAlgebra, DataFrames, Parquet2
include(joinpath(@__DIR__, "src", "CalendarModel.jl"))
using .CalendarModel

const ROOT = normpath(joinpath(@__DIR__, ".."))
const PARQUET = joinpath(ROOT, "data", "processed", "capacity_rpt.parquet")

reps = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 100
B = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 500
out = length(ARGS) >= 3 ? ARGS[3] : joinpath(ROOT, "data", "processed", "sim_coverage_soc.parquet")

D = load_calendar(PARQUET; stage=1, sampling="FF")
names = param_names(D)
x0 = vcat(log(0.02), 30.0, zeros(length(D.levels)), 0.6)
@printf("design: %d rows, %d cells, %d conditions | threads: %d\n", length(D.y), length(D.cell_ids), length(D.conds),
        Threads.nthreads())

fl = fit_nls(D, x0)
truth = fit_mixed(D, fl.theta)
@printf("truth: sigma=%.5f tau=%.4f\n", truth.sigma, truth.tau)

function one_rep(i)
    rng = Xoshiro(1000 + i)
    d = simulate(D, truth.theta, truth.sigma, truth.tau, rng)
    try
        fl = fit_nls(d, truth.theta)
        mm = fit_mixed(d, fl.theta)
        draws = cluster_bootstrap(d, fl.theta, B, rng)
        se_f, se_b = sqrt.(diag(fl.cov)), vec(std(draws; dims=1))
        plo = [quantile(draws[:, j], 0.025) for j in axes(draws, 2)]
        phi = [quantile(draws[:, j], 0.975) for j in axes(draws, 2)]
        return [(rep=i - 1, param=names[j], truth=truth.theta[j], est_nls=fl.theta[j], est_mixed=mm.theta[j],
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
    (k % 10 == 0 || k == reps) && @printf("  %d / %d replicates (%.0f s)\n", k, reps, time() - t_start)
end

rows = reduce(vcat, [r for r in results if r !== nothing]; init=NamedTuple[])
mkpath(dirname(out))
Parquet2.writefile(out, DataFrame(rows))
@printf("saved %d / %d replicates to %s (%.0f s)\n", length(rows) ÷ length(names), reps, out, time() - t_start)
