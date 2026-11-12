"""
Calendar-aging model and estimation utilities (Julia port of `src/calendar_model.py`).

Model (power law in time, Arrhenius in temperature, free effect per SoC level):

    Q_rel(t) = 1 - k * (t^z - t0^z)
    log k    = log_k_ref - (Ea / R) * (1/T - 1/T_ref) + delta[SoC level]     (delta = 0 at SoC 0.5)

Q_rel is normalized to each cell's first regular check-up at time t0; time is in years. The mixed model adds a
cell-level random effect on log k, b_c ~ N(0, tau^2), integrated out by adaptive Gauss-Hermite quadrature.

Parameter vector: theta = [log_k_ref, Ea_kJ, delta_1, ..., delta_L, z].
"""
module CalendarModel

using DataFrames, Parquet2, LinearAlgebra, ForwardDiff, Optim, Random, Statistics

export Design, load_calendar, param_names, fade, fit_nls, fit_mixed, cluster_bootstrap, simulate, subset_rows

const R_GAS = 8.314462618        # J / (mol K)
const T_REF = 296.15             # K (23 degC)
const SOC_REF = 0.5
const REGULAR = ("CU", "exCU")

# Observations in model-ready form. `lvl` indexes the non-reference SoC levels (0 = SoC 0.5).
struct Design
    y::Vector{Float64}           # Q_rel
    t::Vector{Float64}           # years
    t0::Vector{Float64}          # years, first regular check-up of the cell
    arr::Vector{Float64}         # 1/T - 1/T_ref
    lvl::Vector{Int}
    cell::Vector{Int}
    cond::Vector{Int}
    levels::Vector{Float64}
    cell_ids::Vector{String}
    conds::Vector{String}
end

with_y(D::Design, y) = Design(y, D.t, D.t0, D.arr, D.lvl, D.cell, D.cond, D.levels, D.cell_ids, D.conds)

function subset_rows(D::Design, rows)
    Design(D.y[rows], D.t[rows], D.t0[rows], D.arr[rows], D.lvl[rows], D.cell[rows], D.cond[rows],
           D.levels, D.cell_ids, D.conds)
end

param_names(D::Design) = vcat(["log_k_ref", "Ea_kJ"], ["delta_soc" * string(round(s; digits=2)) for s in D.levels], ["z"])

"""Regular check-ups for one calendar arm, as a `Design` (same filters as `load_calendar` in Python)."""
function load_calendar(path; stage=1, sampling="FF")
    cap = DataFrame(Parquet2.Dataset(path); copycols=true)
    keep = (cap.type .== "k") .& (cap.stage .== stage) .& (cap.sampling .== sampling) .&
           in.(cap.kind, Ref(REGULAR)) .& (cap.n_dis_rows .> 0)
    d = cap[keep, :]
    sort!(d, [:cell_id, :days])
    # Q_rel and t0 come from each cell's first regular check-up; that row carries no information and is dropped
    first_idx = Dict{String,Int}()
    for (i, c) in enumerate(d.cell_id)
        haskey(first_idx, c) || (first_idx[c] = i)
    end
    q0 = [d.Q_Ah[first_idx[c]] for c in d.cell_id]
    days0 = [d.days[first_idx[c]] for c in d.cell_id]
    t = d.days ./ 365.25
    t0 = days0 ./ 365.25
    use = t .> t0
    d, q0, t, t0 = d[use, :], q0[use], t[use], t0[use]

    soc = round.(d.soc_max_tp; digits=2)
    levels = sort(filter(!=(SOC_REF), unique(soc)))
    cond_label = string.(Int.(d.amb_temp_tp)) .* "C/" .* [string(round(s; digits=2)) for s in soc]
    # condition labels sorted like the Python "{:.2f}" labels (temperature first, then SoC numerically)
    key = sort(unique(zip(Int.(d.amb_temp_tp), soc)))
    conds = [string(T) * "C/" * string(s) for (T, s) in key]
    cell_ids = sort(unique(d.cell_id))
    return Design(d.Q_Ah ./ q0, t, t0, 1 ./ (d.amb_temp_tp .+ 273.15) .- 1 / T_REF,
                  [s == SOC_REF ? 0 : findfirst(==(s), levels) for s in soc],
                  [searchsortedfirst(cell_ids, c) for c in d.cell_id],
                  [findfirst(==(c), conds) for c in cond_label],
                  levels, cell_ids, conds)
end

"""Fade 1 - Q_rel predicted by the model (generic over the element type, so ForwardDiff can differentiate it)."""
function fade(theta::AbstractVector{T}, D::Design) where {T}
    z = theta[end]
    out = Vector{T}(undef, length(D.y))
    for i in eachindex(out)
        lr = theta[1] - theta[2] * 1e3 / R_GAS * D.arr[i] + (D.lvl[i] == 0 ? zero(T) : theta[2 + D.lvl[i]])
        out[i] = exp(lr) * (D.t[i]^z - D.t0[i]^z)
    end
    return out
end

"""Nonlinear least squares (Levenberg-Marquardt); returns estimate, Fisher (Gauss-Newton) covariance, RSS, n, p."""
function fit_nls(D::Design, theta0; maxit=200)
    yf = 1 .- D.y
    res(th) = yf .- fade(th, D)
    th = float.(theta0)
    r = res(th)
    rss = dot(r, r)
    lambda = 1e-3
    for _ in 1:maxit
        J = ForwardDiff.jacobian(res, th)
        H, g = J' * J, J' * r
        accepted = false
        for _ in 1:40
            step = -((H + lambda * Diagonal(max.(diag(H), 1e-12))) \ g)
            r_new = res(th .+ step)
            rss_new = dot(r_new, r_new)
            if isfinite(rss_new) && rss_new < rss
                th, r, rss_old, rss = th .+ step, r_new, rss, rss_new
                lambda = max(lambda / 3, 1e-12)
                accepted = true
                if rss_old - rss < 1e-13 * rss
                    accepted = false     # converged
                end
                break
            end
            lambda *= 4
        end
        accepted || break
    end
    n, p = length(r), length(th)
    J = ForwardDiff.jacobian(res, th)
    cov = (rss / (n - p)) * inv(J' * J)
    return (theta=th, cov=cov, rss=rss, n=n, p=p, resid=r)
end

# Gauss-Hermite nodes and weights (Golub-Welsch); weights sum to sqrt(pi)
function gauss_hermite(m)
    J = SymTridiagonal(zeros(m), [sqrt(i / 2) for i in 1:m-1])
    e = eigen(J)
    return e.values, sqrt(pi) .* e.vectors[1, :] .^ 2
end

const LOG_SIGMA_BOUNDS = (log(1e-5), log(0.05))
const LOG_TAU_BOUNDS = (log(1e-3), log(3.0))

"""
Negative marginal log-likelihood; cell effect b ~ N(0, tau^2) on log k, integrated by adaptive Gauss-Hermite.

The data are far more informative about b than the prior (likelihood width ~ sigma / (fade sqrt(n)) << tau), so the
nodes are centred at each cell's posterior mode and scaled by its curvature. log sigma and log tau are confined to
a box by a quadratic penalty so that a degenerate fit (tau -> inf, sigma -> 0) cannot run away.
"""
function marginal_nll(par, D::Design, x, w)
    p = length(par) - 2
    ls, lt = par[p+1], par[p+2]
    pen = 1e4 * (max(0.0, ls - LOG_SIGMA_BOUNDS[2])^2 + max(0.0, LOG_SIGMA_BOUNDS[1] - ls)^2 +
                 max(0.0, lt - LOG_TAU_BOUNDS[2])^2 + max(0.0, LOG_TAU_BOUNDS[1] - lt)^2)
    sigma = exp(clamp(ls, LOG_SIGMA_BOUNDS...))
    tau = exp(clamp(lt, LOG_TAU_BOUNDS...))
    f = fade(par[1:p], D)
    yf = 1 .- D.y
    nc = length(D.cell_ids)
    n = length(f)

    # Per-cell posterior mode by Fisher scoring (Gauss-Newton curvature is always negative, so it cannot blow up)
    b = zeros(nc)
    g = zeros(nc)
    info = zeros(nc)
    for _ in 1:100
        fill!(g, 0.0); fill!(info, 0.0)
        for i in 1:n
            c = D.cell[i]
            u = f[i] * exp(b[c])
            g[c] += (yf[i] - u) * u / sigma^2
            info[c] += u^2 / sigma^2
        end
        maxstep = 0.0
        for c in 1:nc
            step = clamp((g[c] - b[c] / tau^2) / (info[c] + 1 / tau^2), -0.5, 0.5)
            b[c] += step
            maxstep = max(maxstep, abs(step))
        end
        maxstep < 1e-10 && break
    end
    fill!(info, 0.0)
    for i in 1:n
        info[D.cell[i]] += (f[i] * exp(b[D.cell[i]]))^2 / sigma^2
    end
    s = 1 ./ sqrt.(info .+ 1 / tau^2)

    m = length(x)
    ll = zeros(nc, m)
    for i in 1:n, k in 1:m
        c = D.cell[i]
        u = f[i] * exp(b[c] + sqrt(2) * s[c] * x[k])
        ll[c, k] += -0.5 * ((yf[i] - u) / sigma)^2 - log(sigma) - 0.5 * log(2pi)
    end
    total = 0.0
    a = zeros(m)
    for c in 1:nc
        for k in 1:m
            node = b[c] + sqrt(2) * s[c] * x[k]
            logprior = -0.5 * (node / tau)^2 - log(tau) - 0.5 * log(2pi)
            # int f(b) db ~ sum_k w_k exp(x_k^2) f(m + sqrt2 s x_k) sqrt2 s
            a[k] = ll[c, k] + logprior + log(w[k]) + x[k]^2 + log(sqrt(2) * s[c])
        end
        amax = maximum(a)
        total += amax + log(sum(exp.(a .- amax)))
    end
    return -total + pen
end

function num_hessian(f, x; h=1e-4)
    p = length(x)
    H = zeros(p, p)
    step = h .* max.(1.0, abs.(x))
    E = Matrix(1.0I, p, p)
    for i in 1:p, j in i:p
        ei, ej = E[:, i] .* step[i], E[:, j] .* step[j]
        H[i, j] = H[j, i] = (f(x + ei + ej) - f(x + ei - ej) - f(x - ei + ej) + f(x - ei - ej)) / (4 * step[i] * step[j])
    end
    return H
end

"""
Nonlinear mixed model with a cell-level random effect on log k (Gaussian noise, ML).

Returns fixed-effect estimates with inverse-Hessian SEs, sigma, tau, and the maximized log-likelihood.
"""
function fit_mixed(D::Design, theta0; tau0=0.1, n_gh=15)
    x, w = gauss_hermite(n_gh)
    f = par -> marginal_nll(par, D, x, w)
    base = fit_nls(D, theta0)
    x0 = vcat(base.theta, log(sqrt(base.rss / (base.n - base.p))), log(tau0))
    r1 = optimize(f, x0, BFGS(), Optim.Options(g_abstol=1e-6, iterations=500))
    r2 = optimize(f, Optim.minimizer(r1), NelderMead(), Optim.Options(iterations=5000))
    par = Optim.minimizer(r2)
    p = length(theta0)
    se = try
        sqrt.(abs.(diag(inv(num_hessian(f, par)))))[1:p]
    catch
        fill(NaN, p)
    end
    return (theta=par[1:p], se=se, sigma=exp(par[p+1]), tau=exp(par[p+2]), loglik=-Optim.minimum(r2), par=par)
end

"""Resample cells with replacement within each condition and refit; returns the successful draws as a matrix."""
function cluster_bootstrap(D::Design, theta_hat, B, rng)
    rows_by_cell = [Int[] for _ in D.cell_ids]
    for (i, c) in enumerate(D.cell) push!(rows_by_cell[c], i) end
    cells_by_cond = [Int[] for _ in D.conds]
    for c in eachindex(rows_by_cell)
        push!(cells_by_cond[D.cond[rows_by_cell[c][1]]], c)
    end
    draws = Vector{Vector{Float64}}()
    for _ in 1:B
        rows = Int[]
        for cells in cells_by_cond, _ in cells
            append!(rows, rows_by_cell[rand(rng, cells)])
        end
        try
            push!(draws, fit_nls(subset_rows(D, rows), theta_hat).theta)
        catch
            continue
        end
    end
    return reduce(hcat, draws)'
end

"""Draw cell effects b_c ~ N(0, tau^2) and noise ~ N(0, sigma^2) on the observed design."""
function simulate(D::Design, theta, sigma, tau, rng)
    b = tau .* randn(rng, length(D.cell_ids))
    yf = fade(theta, D) .* exp.(b[D.cell])
    return with_y(D, 1 .- yf .+ sigma .* randn(rng, length(yf)))
end

end # module
