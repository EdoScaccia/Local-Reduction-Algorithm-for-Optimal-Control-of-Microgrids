using JLD2
using Random
using Statistics

include("load_problem_data.jl")

# User inputs.
formulation_input = "MIP"  # "NLP" or "MIP"
n_monte_carlo_scenarios = 100_000
selected_solution_idx = nothing
monte_carlo_seed = 1234

const DEFAULT_SOC_TAGS = ["SOC020", "SOC040", "SOC060", "SOC080", "SOC0100"]
const DEFAULT_PG_TAGS = ["PG20", "PG40", "PG60", "PG80", "PG100"]
const VIOLATION_TOL = 1e-7
const POWER_ZERO_TOL = 1e-6

function _last_finite_col(matrix)
    for j in size(matrix, 2):-1:1
        all(isfinite, matrix[:, j]) && return j
    end
    error("Could not find a finite solution column in the saved result.")
end

function _extract_solution_from_result(result_data; solution_idx=nothing)
    haskey(result_data, "master_x_obj_history") || error("Result file does not contain `master_x_obj_history`.")
    haskey(result_data, "u_star_history") || error("Result file does not contain `u_star_history`.")

    u_star_history = result_data["u_star_history"]
    master_x_obj_history = result_data["master_x_obj_history"]
    selected_idx = solution_idx === nothing ? _last_finite_col(u_star_history) : Int(solution_idx)

    selected_idx >= 1 && selected_idx <= size(u_star_history, 2) || error("solution_idx=$selected_idx is outside 1:$(size(u_star_history, 2)).")
    all(isfinite, u_star_history[:, selected_idx]) || error("u_star_history column $selected_idx is not finite.")
    isfinite(master_x_obj_history[selected_idx]) || error("master_x_obj_history[$selected_idx] is not finite.")

    return u_star_history[:, selected_idx], master_x_obj_history[selected_idx], selected_idx
end

function _result_value(result_data, key; default=NaN)
    return haskey(result_data, key) ? Float64(result_data[key]) : default
end

function _result_field(result_data, key; default=NaN)
    return haskey(result_data, key) ? result_data[key] : default
end

function _microgrid_value(microgrid_params, key, aliases...)
    if haskey(microgrid_params, key)
        return microgrid_params[key]
    end
    for alias in aliases
        if haskey(microgrid_params, alias)
            return microgrid_params[alias]
        end
    end
    error("Could not find microgrid parameter `$key`.")
end

function _param_value(result_data, microgrid_params, key; aliases=String[], default=nothing)
    if haskey(result_data, key)
        return Float64(result_data[key])
    end
    if haskey(microgrid_params, key)
        return Float64(microgrid_params[key])
    end
    if haskey(result_data, "data_config")
        data_config = result_data["data_config"]
        key_sym = Symbol(key)
        if hasproperty(data_config, key_sym)
            return Float64(getproperty(data_config, key_sym))
        end
    end

    for alias in aliases
        if haskey(result_data, alias)
            return Float64(result_data[alias])
        end
        if haskey(microgrid_params, alias)
            return Float64(microgrid_params[alias])
        end
        if haskey(result_data, "data_config")
            data_config = result_data["data_config"]
            alias_sym = Symbol(alias)
            if hasproperty(data_config, alias_sym)
                return Float64(getproperty(data_config, alias_sym))
            end
        end
    end

    default === nothing && error("Could not find parameter `$key`.")
    return Float64(default)
end

function _load_bounds_and_params(result_data)
    if haskey(result_data, "uncertainty_bounds") && haskey(result_data, "microgrid_params")
        return result_data["uncertainty_bounds"], result_data["microgrid_params"], get(result_data, "resolved_data_path", "")
    end

    resolved_data_path = get(result_data, "resolved_data_path", "")
    if isempty(resolved_data_path)
        _, uncertainty_bounds, _, _, _, _, microgrid_params = read_microgrid_problem_data()
    else
        _, uncertainty_bounds, _, _, _, _, microgrid_params = read_microgrid_problem_data(resolved_data_path)
    end
    return uncertainty_bounds, microgrid_params, resolved_data_path
end

function _sample_uniform_scenarios(rng, uncertainty_bounds, n_scenarios, grid_size)
    n_t = grid_size - 1
    samples = Dict{String, Matrix{Float64}}()

    specs = (
        ("PL", "PL_lb", "PL_ub"),
        ("PR", "PR_lb", "PR_ub"),
        ("C_buy", "C_buy_lb", "C_buy_ub"),
        ("C_sell", "C_sell_lb", "C_sell_ub"),
        ("eta_ch", "eta_ch_lb", "eta_ch_ub"),
        ("rho_dis", "eta_dis_lb", "eta_dis_ub"),
    )

    for (name, lb_key, ub_key) in specs
        A = Matrix{Float64}(undef, n_t, n_scenarios)
        for t in 1:n_t
            lb = uncertainty_bounds[lb_key][t]
            ub = uncertainty_bounds[ub_key][t]
            A[t, :] .= rand(rng, n_scenarios) .* (ub - lb) .+ lb
        end
        samples[name] = A
    end

    return samples
end

function _unpack_u_star(u_star, grid_size)
    n_t = grid_size - 1
    u_dim = div(length(u_star), n_t)
    length(u_star) == u_dim * n_t || error("u_star length $(length(u_star)) is incompatible with grid_size=$grid_size.")
    u_dim == 2 || error("Expected one-battery u_dim=2, found u_dim=$u_dim.")

    Pc = [u_star[u_dim * (t - 1) + 1] for t in 1:n_t]
    Pd = [u_star[u_dim * (t - 1) + 2] for t in 1:n_t]
    return Pc, Pd
end

function _zero_small_power(v; power_tol::Float64=POWER_ZERO_TOL)
    return abs(v) <= power_tol ? 0.0 : v
end

function _propagate_soc(Pc, Pd, eta_ch, rho_dis, microgrid_params, n_star)
    n_t, n_scenarios = size(eta_ch)
    soc = Matrix{Float64}(undef, n_t + 1, n_scenarios)
    energy = Matrix{Float64}(undef, n_t + 1, n_scenarios)

    soc0 = _microgrid_value(microgrid_params, "SoC0")
    delta_t = _microgrid_value(microgrid_params, "Δt", "Î”t")
    e_batt = _microgrid_value(microgrid_params, "E_BATT")
    capacity = n_star * e_batt
    capacity > 0 || error("n_star gives non-positive battery capacity: n_star=$n_star.")

    energy[1, :] .= soc0 * capacity
    soc[1, :] .= soc0
    for t in 1:n_t
        energy[t + 1, :] .= energy[t, :] .+ (eta_ch[t, :] .* Pc[t] .- rho_dis[t, :] .* Pd[t]) .* delta_t
        soc[t + 1, :] .= energy[t + 1, :] ./ capacity
    end

    return soc
end

function _scenario_count(mask)
    return vec(sum(mask; dims=1))
end

function _safe_mean(v)
    isempty(v) && return NaN
    return mean(v)
end

function _safe_max(v)
    isempty(v) && return NaN
    return maximum(v)
end

function _fmt(x; digits=4)
    if x isa Integer
        return string(x)
    elseif x isa AbstractFloat
        isfinite(x) || return string(x)
        return string(round(x; digits=digits))
    else
        return string(x)
    end
end

function _latest_result_for_case(results_dir, formulation, soc_tag, pg_tag)
    isdir(results_dir) || error("Results directory does not exist: $results_dir")
    prefix = uppercase(formulation) * "_" * soc_tag * "_H1_" * pg_tag * "_"
    candidates = filter(
        path -> startswith(basename(path), prefix) && endswith(path, ".jld2"),
        readdir(results_dir; join=true),
    )
    isempty(candidates) && return nothing
    return first(sort(candidates; by=mtime, rev=true))
end

function find_monte_carlo_result_files(
    formulation::AbstractString;
    results_dir::AbstractString=joinpath(@__DIR__, "Results"),
    soc_tags=DEFAULT_SOC_TAGS,
    pg_tags=DEFAULT_PG_TAGS,
)
    formulation_uc = uppercase(formulation)
    formulation_uc in ("NLP", "MIP", "MILP") || error("formulation must be NLP, MIP, or MILP.")
    file_formulation = formulation_uc == "MILP" ? "MIP" : formulation_uc

    files = NamedTuple[]
    missing = NamedTuple[]
    for soc_tag in soc_tags
        for pg_tag in pg_tags
            path = _latest_result_for_case(results_dir, file_formulation, soc_tag, pg_tag)
            if path === nothing
                push!(missing, (soc=soc_tag, pg=pg_tag))
            else
                push!(files, (soc=soc_tag, pg=pg_tag, path=path))
            end
        end
    end

    return files, missing
end

function run_monte_carlo_from_result(
    result_path::AbstractString;
    n_scenarios::Int=100_000,
    solution_idx=nothing,
    rng=MersenneTwister(1234),
    violation_tol::Float64=VIOLATION_TOL,
    power_zero_tol::Float64=POWER_ZERO_TOL,
    log_label::AbstractString="",
    log_every::Int=10_000,
)
    result_data = JLD2.load(result_path)
    uncertainty_bounds, microgrid_params, resolved_data_path = _load_bounds_and_params(result_data)
    u_star, gamma_star, selected_idx = _extract_solution_from_result(result_data; solution_idx=solution_idx)

    n_star = _result_value(result_data, "n_star")
    s_star = _result_value(result_data, "s_star")
    isfinite(n_star) || error("Result file does not contain finite `n_star`.")
    isfinite(s_star) || error("Result file does not contain finite `s_star`.")

    grid_size = _microgrid_value(microgrid_params, "grid_size")
    n_t = grid_size - 1
    Pc, Pd = _unpack_u_star(u_star, grid_size)
    Pc .= _zero_small_power.(Pc; power_tol=power_zero_tol)
    Pd .= _zero_small_power.(Pd; power_tol=power_zero_tol)
    soc_min = _microgrid_value(microgrid_params, "SOC_MIN")
    soc_max = _microgrid_value(microgrid_params, "SOC_MAX")
    pg_plus_max = _microgrid_value(microgrid_params, "PG_plus_MAX")
    pg_minus_max = _microgrid_value(microgrid_params, "PG_minus_MAX")
    delta_t = _microgrid_value(microgrid_params, "Δt", "Î”t")
    e_batt = _microgrid_value(microgrid_params, "E_BATT")
    battery_cost = _param_value(result_data, microgrid_params, "battery_cost"; aliases=["Battery_Cost"], default=200.0)
    pv_cost = _param_value(result_data, microgrid_params, "pv_cost"; aliases=["PV_Cost"], default=1500.0)
    base_pv_capacity_kw = _param_value(result_data, microgrid_params, "base_pv_capacity_kw"; aliases=["Base_PV_Capacity_kW"], default=90.84)
    investment_cost = n_star * e_batt * battery_cost + s_star * base_pv_capacity_kw * pv_cost

    Pc_pos = max.(Pc, 0.0)
    Pd_pos = max.(Pd, 0.0)
    battery_logic_violation = min.(Pc_pos, Pd_pos)
    battery_logic_count = count(battery_logic_violation .> power_zero_tol)
    grid_logic_count = 0
    logic_count_per_scenario = battery_logic_count + grid_logic_count
    logic_max = maximum(vcat(battery_logic_violation, 0.0))

    soc_constraint_count = 2 * grid_size
    grid_constraint_count = 2 * n_t
    logic_constraint_count = 2 * n_t
    cost_constraint_count = 1
    total_constraint_count = soc_constraint_count + grid_constraint_count + logic_constraint_count + cost_constraint_count

    n_infeasible = 0
    violated_count_sum = 0.0
    soc_count_sum = 0.0
    grid_count_sum = 0.0
    logic_count_sum = 0.0
    cost_count_sum = 0.0
    max_violation_sum = 0.0
    max_violation_seen = 0.0

    chunk_size = log_every > 0 ? log_every : n_scenarios
    n_done = 0
    while n_done < n_scenarios
        n_chunk = min(chunk_size, n_scenarios - n_done)
        samples = _sample_uniform_scenarios(rng, uncertainty_bounds, n_chunk, grid_size)

        PL = samples["PL"]
        PR = samples["PR"]
        C_buy = samples["C_buy"]
        C_sell = samples["C_sell"]
        eta_ch = samples["eta_ch"]
        rho_dis = samples["rho_dis"]

        Pc_mat = repeat(reshape(Pc, n_t, 1), 1, n_chunk)
        Pd_mat = repeat(reshape(Pd, n_t, 1), 1, n_chunk)
        required_grid_net = Pc_mat .+ PL .- s_star .* PR .- Pd_mat

        soc = _propagate_soc(Pc, Pd, eta_ch, rho_dis, microgrid_params, n_star)
        soc_lb_violation = max.(soc_min .- soc, 0.0)
        soc_ub_violation = max.(soc .- soc_max, 0.0)
        soc_counts = _scenario_count(soc_lb_violation .> violation_tol) .+
            _scenario_count(soc_ub_violation .> violation_tol)
        soc_max_by_scenario = vec(maximum(max.(soc_lb_violation, soc_ub_violation); dims=1))

        grid_plus_violation = max.(required_grid_net .- pg_plus_max, 0.0)
        grid_minus_violation = max.(-required_grid_net .- pg_minus_max, 0.0)
        grid_counts = _scenario_count(grid_plus_violation .> violation_tol) .+
            _scenario_count(grid_minus_violation .> violation_tol)
        grid_max_by_scenario = vec(maximum(max.(grid_plus_violation, grid_minus_violation); dims=1))

        logic_counts = fill(Float64(logic_count_per_scenario), n_chunk)
        logic_max_by_scenario = fill(logic_max, n_chunk)

        PG_plus = max.(required_grid_net, 0.0)
        PG_minus = max.(-required_grid_net, 0.0)
        operating_cost = vec(sum(delta_t .* (C_buy .* PG_plus .- C_sell .* PG_minus); dims=1))
        cost_violation_by_scenario = max.(investment_cost .+ operating_cost .- gamma_star, 0.0)
        cost_counts = Float64.(cost_violation_by_scenario .> violation_tol)

        violated_counts = soc_counts .+ grid_counts .+ logic_counts .+ cost_counts
        max_violation_by_scenario = max.(soc_max_by_scenario, grid_max_by_scenario, logic_max_by_scenario, cost_violation_by_scenario)

        feasibility_counts = soc_counts .+ grid_counts .+ logic_counts
        n_infeasible += count(feasibility_counts .> 0)
        violated_count_sum += sum(violated_counts)
        soc_count_sum += sum(soc_counts)
        grid_count_sum += sum(grid_counts)
        logic_count_sum += sum(logic_counts)
        cost_count_sum += sum(cost_counts)
        max_violation_sum += sum(max_violation_by_scenario)
        max_violation_seen = max(max_violation_seen, maximum(max_violation_by_scenario))

        n_done += n_chunk
        if log_every > 0 && (n_done % log_every == 0 || n_done == n_scenarios)
            prefix = isempty(log_label) ? "Monte Carlo" : log_label
            println(prefix, ": ", n_done, " / ", n_scenarios, " scenarios")
        end
    end

    total_violated = soc_count_sum + grid_count_sum + logic_count_sum + cost_count_sum

    runtime = Float64(_result_field(result_data, "total_elapsed_seconds"; default=NaN))

    return Dict(
        "result_path" => String(result_path),
        "resolved_data_path" => String(resolved_data_path),
        "solution_index" => selected_idx,
        "n_scenarios" => n_scenarios,
        "runtime_seconds" => runtime,
        "n_star" => n_star,
        "s_star" => s_star,
        "gamma_star" => gamma_star,
        "feasible_scenarios_pct" => 100.0 * (1.0 - n_infeasible / n_scenarios),
        "infeasible_scenarios_pct" => 100.0 * n_infeasible / n_scenarios,
        "constraint_violated_to_total_pct" => 100.0 * violated_count_sum / (n_scenarios * total_constraint_count),
        "soc_bounds_share_of_violated_pct" => total_violated == 0 ? 0.0 : 100.0 * soc_count_sum / total_violated,
        "logic_share_of_violated_pct" => total_violated == 0 ? 0.0 : 100.0 * logic_count_sum / total_violated,
        "grid_cap_share_of_violated_pct" => total_violated == 0 ? 0.0 : 100.0 * grid_count_sum / total_violated,
        "cost_share_of_violated_pct" => total_violated == 0 ? 0.0 : 100.0 * cost_count_sum / total_violated,
        "average_max_constraint_violation" => max_violation_sum / n_scenarios,
        "max_constraint_violation" => max_violation_seen,
        "total_constraint_count" => total_constraint_count,
        "soc_constraint_count" => soc_constraint_count,
        "logic_constraint_count" => logic_constraint_count,
        "grid_constraint_count" => grid_constraint_count,
        "cost_constraint_count" => cost_constraint_count,
        "n_infeasible" => n_infeasible,
    )
end

function _aggregate_mc_summaries(formulation, summaries, failures, missing, n_scenarios)
    runtimes = [s["runtime_seconds"] for s in summaries if isfinite(s["runtime_seconds"])]
    return Dict(
        "formulation" => uppercase(formulation) == "MILP" ? "MIP" : uppercase(formulation),
        "n_requested_scenarios_per_result" => n_scenarios,
        "n_result_files_evaluated" => length(summaries),
        "n_failed_result_files" => length(failures),
        "n_missing_result_files" => length(missing),
        "average_runtime_seconds" => _safe_mean(runtimes),
        "max_runtime_seconds" => _safe_max(runtimes),
        "average_gamma_star" => _safe_mean([s["gamma_star"] for s in summaries]),
        "average_feasible_scenarios_pct" => _safe_mean([s["feasible_scenarios_pct"] for s in summaries]),
        "average_infeasible_scenarios_pct" => _safe_mean([s["infeasible_scenarios_pct"] for s in summaries]),
        "average_constraint_violated_to_total_problem_constraints_pct" => _safe_mean([s["constraint_violated_to_total_pct"] for s in summaries]),
        "among_violated_average_soc_bounds_pct" => _safe_mean([s["soc_bounds_share_of_violated_pct"] for s in summaries]),
        "among_violated_average_logic_constraints_pct" => _safe_mean([s["logic_share_of_violated_pct"] for s in summaries]),
        "among_violated_average_grid_cap_constraints_pct" => _safe_mean([s["grid_cap_share_of_violated_pct"] for s in summaries]),
        "among_violated_average_cost_constraints_pct" => _safe_mean([s["cost_share_of_violated_pct"] for s in summaries]),
        "average_maximum_constraint_violation" => _safe_mean([s["max_constraint_violation"] for s in summaries]),
        "average_scenario_max_constraint_violation" => _safe_mean([s["average_max_constraint_violation"] for s in summaries]),
        "per_result_summaries" => summaries,
        "failed_result_files" => failures,
        "missing_cases" => missing,
    )
end

function _print_aggregate_mc_summary(summary)
    println()
    println("Monte Carlo aggregate summary")
    println(repeat("-", 96))
    rows = [
        ("Formulation", summary["formulation"]),
        ("Result files evaluated", summary["n_result_files_evaluated"]),
        ("Missing result files", summary["n_missing_result_files"]),
        ("Failed result files", summary["n_failed_result_files"]),
        ("Scenarios per result", summary["n_requested_scenarios_per_result"]),
        ("Average runtime (s)", _fmt(summary["average_runtime_seconds"])),
        ("Max runtime (s)", _fmt(summary["max_runtime_seconds"])),
        ("Average gamma*", _fmt(summary["average_gamma_star"])),
        ("Average feasible scenarios (%)", _fmt(summary["average_feasible_scenarios_pct"])),
        ("Average infeasible scenarios (%)", _fmt(summary["average_infeasible_scenarios_pct"])),
        ("Average constraint violated / total constraints (%)", _fmt(summary["average_constraint_violated_to_total_problem_constraints_pct"])),
        ("Among violated: average SoC bounds (%)", _fmt(summary["among_violated_average_soc_bounds_pct"])),
        ("Among violated: average logic constraints (%)", _fmt(summary["among_violated_average_logic_constraints_pct"])),
        ("Among violated: average grid cap constraints (%)", _fmt(summary["among_violated_average_grid_cap_constraints_pct"])),
        ("Among violated: average cost constraints (%)", _fmt(summary["among_violated_average_cost_constraints_pct"])),
        ("Average maximum constraint violation", _fmt(summary["average_maximum_constraint_violation"])),
    ]

    key_width = maximum(length(String(row[1])) for row in rows)
    for (label, value) in rows
        println(rpad(String(label), key_width), " | ", value)
    end

    if !isempty(summary["missing_cases"])
        println()
        println("Missing cases")
        println(repeat("-", 96))
        for case in summary["missing_cases"]
            println(case.soc, " ", case.pg)
        end
    end

    if !isempty(summary["failed_result_files"])
        println()
        println("Failed result files")
        println(repeat("-", 96))
        for failure in summary["failed_result_files"]
            println(basename(failure.path), " | ", failure.error)
        end
    end
end

function run_monte_carlo_sweep(
    formulation::AbstractString;
    n_scenarios::Int=100_000,
    results_dir::AbstractString=joinpath(@__DIR__, "Results"),
    soc_tags=DEFAULT_SOC_TAGS,
    pg_tags=DEFAULT_PG_TAGS,
    solution_idx=nothing,
    seed::Int=1234,
    print_summary::Bool=true,
)
    files, missing = find_monte_carlo_result_files(
        formulation;
        results_dir=results_dir,
        soc_tags=soc_tags,
        pg_tags=pg_tags,
    )

    summaries = Dict{String, Any}[]
    failures = NamedTuple[]
    rng = MersenneTwister(seed)

    for file in files
        log_label = "Running Monte Carlo for " * file.pg * " " * file.soc
        println(log_label, ":")
        try
            result_summary = run_monte_carlo_from_result(
                file.path;
                n_scenarios=n_scenarios,
                solution_idx=solution_idx,
                rng=rng,
                log_label=log_label,
            )
            result_summary["soc_case"] = file.soc
            result_summary["pg_case"] = file.pg
            push!(summaries, result_summary)
        catch err
            push!(failures, (soc=file.soc, pg=file.pg, path=file.path, error=sprint(showerror, err)))
        end
    end

    aggregate = _aggregate_mc_summaries(formulation, summaries, failures, missing, n_scenarios)
    print_summary && _print_aggregate_mc_summary(aggregate)
    return aggregate
end

mc_summary = run_monte_carlo_sweep(
        formulation_input;
        n_scenarios=n_monte_carlo_scenarios,
        solution_idx=selected_solution_idx,
        seed=monte_carlo_seed,
    )
