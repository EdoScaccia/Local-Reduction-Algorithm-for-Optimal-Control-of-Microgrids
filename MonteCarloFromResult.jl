using JLD2
using Statistics

const PYPLOT_AVAILABLE = Ref(false)

function _ensure_pyplot!()
    PYPLOT_AVAILABLE[] && return true
    try
        @eval using PyPlot
        PYPLOT_AVAILABLE[] = true
        return true
    catch err
        @warn "PyPlot is not available; Monte Carlo tables and deterministic checks will still run, but plotting is disabled." reason=sprint(showerror, err)
        return false
    end
end

function _ensure_problem_data_loader!()
    isdefined(@__MODULE__, :read_microgrid_problem_data) && return
    include("load_problem_data.jl")
end

result_path = joinpath(@__DIR__, "Results", "MILP_SOC060_H1_PG100_B1_2026-04-28_123251.jld2")

# Leave as `nothing` to use the last finite saved solution.
# Set to an integer, e.g. `selected_solution_idx = 3`, to replay a specific
# column of `u_star_history` and `master_x_obj_history`.
selected_solution_idx = nothing

const VALIDATION_TOL = 1e-7

function _last_finite_col(matrix)
    for j in size(matrix, 2):-1:1
        any(isfinite, matrix[:, j]) && return j
    end
    error("Could not find a finite solution column in the saved result.")
end

function _extract_solution_from_result(result_data; solution_idx=nothing)
    haskey(result_data, "master_x_obj_history") || error("Result file does not contain `master_x_obj_history`.")
    haskey(result_data, "u_star_history") || error("Result file does not contain `u_star_history`.")

    u_star_history = result_data["u_star_history"]
    master_x_obj_history = result_data["master_x_obj_history"]

    selected_idx = solution_idx === nothing ? _last_finite_col(u_star_history) : Int(solution_idx)
    selected_idx >= 1 && selected_idx <= size(u_star_history, 2) || error("solution_idx=$selected_idx is outside the saved history range 1:$(size(u_star_history, 2)).")
    any(isfinite, u_star_history[:, selected_idx]) || error("u_star_history column $selected_idx is not finite.")
    isfinite(master_x_obj_history[selected_idx]) || error("master_x_obj_history[$selected_idx] is not finite.")

    u_star = u_star_history[:, selected_idx]
    gamma_star = master_x_obj_history[selected_idx]

    return u_star, gamma_star, selected_idx
end

function _sample_uniform_scenarios(uncertainty_bounds, n_scenarios, grid_size)
    n_t = grid_size - 1

    PL = Matrix{Float64}(undef, n_t, n_scenarios)
    PR = Matrix{Float64}(undef, n_t, n_scenarios)
    C_buy = Matrix{Float64}(undef, n_t, n_scenarios)
    C_sell = Matrix{Float64}(undef, n_t, n_scenarios)
    eta_ch = Matrix{Float64}(undef, n_t, n_scenarios)
    rho_dis = Matrix{Float64}(undef, n_t, n_scenarios)

    for t in 1:n_t
        PL[t, :] .= rand(n_scenarios) .* (uncertainty_bounds["PL_ub"][t] - uncertainty_bounds["PL_lb"][t]) .+ uncertainty_bounds["PL_lb"][t]
        PR[t, :] .= rand(n_scenarios) .* (uncertainty_bounds["PR_ub"][t] - uncertainty_bounds["PR_lb"][t]) .+ uncertainty_bounds["PR_lb"][t]
        C_buy[t, :] .= rand(n_scenarios) .* (uncertainty_bounds["C_buy_ub"][t] - uncertainty_bounds["C_buy_lb"][t]) .+ uncertainty_bounds["C_buy_lb"][t]
        C_sell[t, :] .= rand(n_scenarios) .* (uncertainty_bounds["C_sell_ub"][t] - uncertainty_bounds["C_sell_lb"][t]) .+ uncertainty_bounds["C_sell_lb"][t]
        eta_ch[t, :] .= rand(n_scenarios) .* (uncertainty_bounds["eta_ch_ub"][t] - uncertainty_bounds["eta_ch_lb"][t]) .+ uncertainty_bounds["eta_ch_lb"][t]
        rho_dis[t, :] .= rand(n_scenarios) .* (uncertainty_bounds["rho_dis_ub"][t] - uncertainty_bounds["rho_dis_lb"][t]) .+ uncertainty_bounds["rho_dis_lb"][t]
    end

    return Dict("PL" => PL, "PR" => PR, "C_buy" => C_buy, "C_sell" => C_sell, "eta_ch" => eta_ch, "rho_dis" => rho_dis)
end

function _global_corner_scenarios(uncertainty_bounds, grid_size)
    n_t = grid_size - 1
    corners = vec(collect(Iterators.product((0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1))))
    n_corners = length(corners)

    PL = Matrix{Float64}(undef, n_t, n_corners)
    PR = Matrix{Float64}(undef, n_t, n_corners)
    C_buy = Matrix{Float64}(undef, n_t, n_corners)
    C_sell = Matrix{Float64}(undef, n_t, n_corners)
    eta_ch = Matrix{Float64}(undef, n_t, n_corners)
    rho_dis = Matrix{Float64}(undef, n_t, n_corners)

    for (j, corner) in enumerate(corners)
        pl_sel, pr_sel, cb_sel, cs_sel, eta_ch_sel, rho_dis_sel = corner
        PL[:, j] .= pl_sel == 0 ? uncertainty_bounds["PL_lb"] : uncertainty_bounds["PL_ub"]
        PR[:, j] .= pr_sel == 0 ? uncertainty_bounds["PR_lb"] : uncertainty_bounds["PR_ub"]
        C_buy[:, j] .= cb_sel == 0 ? uncertainty_bounds["C_buy_lb"] : uncertainty_bounds["C_buy_ub"]
        C_sell[:, j] .= cs_sel == 0 ? uncertainty_bounds["C_sell_lb"] : uncertainty_bounds["C_sell_ub"]
        eta_ch[:, j] .= eta_ch_sel == 0 ? uncertainty_bounds["eta_ch_lb"] : uncertainty_bounds["eta_ch_ub"]
        rho_dis[:, j] .= rho_dis_sel == 0 ? uncertainty_bounds["rho_dis_lb"] : uncertainty_bounds["rho_dis_ub"]
    end

    return Dict(
        "PL" => PL,
        "PR" => PR,
        "C_buy" => C_buy,
        "C_sell" => C_sell,
        "eta_ch" => eta_ch,
        "rho_dis" => rho_dis,
        "n_corners" => n_corners,
        "corner_flags" => trues(n_corners),
        "corner_labels" => String[string(corner) for corner in corners],
    )
end

function _result_value(result_data, key; default=NaN)
    return haskey(result_data, key) ? Float64(result_data[key]) : default
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

    default === nothing && error("Could not find parameter `$key` in result data or microgrid parameters.")
    return Float64(default)
end

function _unpack_u_star(u_star, grid_size; two_batt=nothing)
    n_t = grid_size - 1
    u_dim = div(length(u_star), n_t)
    length(u_star) == u_dim * n_t || error("u_star length $(length(u_star)) is not compatible with grid_size=$grid_size.")

    if two_batt === nothing
        two_batt = u_dim == 4
    end
    expected_u_dim = two_batt ? 4 : 2
    u_dim == expected_u_dim || error("Expected u_dim=$expected_u_dim, found u_dim=$u_dim.")

    Pc1 = [u_star[u_dim * (t - 1) + 1] for t in 1:n_t]
    Pd1 = [u_star[u_dim * (t - 1) + 2] for t in 1:n_t]
    Pc2 = two_batt ? [u_star[u_dim * (t - 1) + 3] for t in 1:n_t] : zeros(n_t)
    Pd2 = two_batt ? [u_star[u_dim * (t - 1) + 4] for t in 1:n_t] : zeros(n_t)

    return Pc1, Pd1, Pc2, Pd2
end

function _propagate_soc(Pc1_mat, Pd1_mat, Pc2_mat, Pd2_mat, eta_ch, rho_dis, microgrid_params, n_star, m_star; two_batt=false)
    n_t, n_scenarios = size(Pc1_mat)
    energy1 = Matrix{Float64}(undef, n_t + 1, n_scenarios)
    energy2 = Matrix{Float64}(undef, n_t + 1, n_scenarios)
    soc1 = Matrix{Float64}(undef, n_t + 1, n_scenarios)
    soc2 = Matrix{Float64}(undef, n_t + 1, n_scenarios)

    soc0 = microgrid_params["SoC0"]
    delta_t = microgrid_params["Δt"]
    e_batt = microgrid_params["E_BATT"]
    capacity1 = n_star * e_batt
    capacity2 = two_batt ? m_star * e_batt : NaN
    capacity1 > 0 || error("n_star gives non-positive capacity: n_star=$n_star.")
    !two_batt || capacity2 > 0 || error("m_star gives non-positive capacity: m_star=$m_star.")

    energy1[1, :] .= soc0 * capacity1
    energy2[1, :] .= two_batt ? soc0 * capacity2 : NaN
    soc1[1, :] .= soc0
    soc2[1, :] .= two_batt ? soc0 : NaN

    for t in 1:n_t
        energy1[t + 1, :] .= energy1[t, :] .+ (eta_ch[t, :] .* Pc1_mat[t, :] .- rho_dis[t, :] .* Pd1_mat[t, :]) .* delta_t
        soc1[t + 1, :] .= energy1[t + 1, :] ./ capacity1
        if two_batt
            energy2[t + 1, :] .= energy2[t, :] .+ (eta_ch[t, :] .* Pc2_mat[t, :] .- rho_dis[t, :] .* Pd2_mat[t, :]) .* delta_t
            soc2[t + 1, :] .= energy2[t + 1, :] ./ capacity2
        end
    end

    return soc1, soc2, energy1, energy2
end

function _scenario_max(values)
    return vec(maximum(values; dims=1))
end

function _stats_value(stats, key; default=0.0)
    stats === nothing && return default
    haskey(stats, key) && return stats[key]
    string_key = String(key)
    haskey(stats, string_key) && return stats[string_key]
    return default
end

function _result_field(result_data, key; default=NaN)
    haskey(result_data, key) && return result_data[key]
    return default
end

function _scenario_count(mask)
    return vec(sum(mask; dims=1))
end

function _constraint_percent(counts, total_constraints)
    total_constraints <= 0 && return zeros(length(counts))
    return 100.0 .* counts ./ total_constraints
end

function _mean_on_mask(values, mask)
    any(mask) || return 0.0
    return mean(values[mask])
end

function _fmt(x; digits=4)
    if x isa Integer
        return string(x)
    elseif x isa AbstractFloat
        return string(round(x; digits=digits))
    else
        return string(x)
    end
end

function _print_table(title, rows)
    println()
    println(title)
    println(repeat("-", 88))
    key_width = max(10, maximum(length(String(row[1])) for row in rows))
    for (label, value) in rows
        println(rpad(String(label), key_width), " | ", value)
    end
end

function _print_mc_results_tables(summary)
    _print_table("Monte Carlo run summary", [
        ("Formulation", summary["formulation"]),
        ("Local red runtime (s)", _fmt(summary["local_reduction_runtime_seconds"])),
        ("MILP/NLP solved", _fmt(summary["optimizer_solve_count"]; digits=0)),
        ("Solution index", _fmt(summary["solution_index"]; digits=0)),
        ("Last finite solution index", _fmt(summary["last_solution_index"]; digits=0)),
        ("gamma_star", _fmt(summary["gamma_star"])),
        ("Operating cost dt factor", _fmt(summary["operating_cost_dt_factor"])),
        ("Robust infeasible scenarios (%)", _fmt(100 * summary["infeasible_rate"])),
        ("Physical infeasible scenarios (%)", _fmt(100 * summary["physical_infeasible_rate"])),
        ("Max overall violation (mixed units)", _fmt(summary["max_overall_violation"])),
        ("Max constraints violated (%)", _fmt(summary["max_constraint_violation_pct"])),
        ("Avg constraints violated (%)", _fmt(summary["avg_constraint_violation_pct"])),
    ])

    _print_table("Average time/row percentage violated within infeasible scenarios", [
        ("SoC bounds (%)", _fmt(summary["avg_soc_bound_constraint_violation_pct_infeasible"])),
        ("Grid bounds (%)", _fmt(summary["avg_load_balance_constraint_violation_pct_infeasible"])),
        ("Terminal cycle (%)", _fmt(summary["avg_terminal_soc_constraint_violation_pct_infeasible"])),
        ("Epigraph (%)", _fmt(summary["avg_epigraph_constraint_violation_pct_infeasible"])),
    ])

    _print_table("Share of infeasible scenarios with each violation type", [
        ("SoC bounds (%)", _fmt(100 * summary["infeasible_scenario_soc_bound_share"])),
        ("Grid bounds (%)", _fmt(100 * summary["infeasible_scenario_load_balance_share"])),
        ("Terminal cycle (%)", _fmt(100 * summary["infeasible_scenario_terminal_soc_share"])),
        ("Epigraph (%)", _fmt(100 * summary["infeasible_scenario_epigraph_share"])),
        ("SoC and grid bounds (%)", _fmt(100 * summary["infeasible_scenario_soc_and_load_share"])),
    ])

    _print_table("Maximum violation magnitude by type", [
        ("SoC bounds (p.u.)", _fmt(summary["max_soc_bound_violation"])),
        ("Grid bounds (kW)", _fmt(summary["max_load_balance_violation"])),
        ("Epigraph (cost units)", _fmt(summary["max_epigraph_violation"])),
        ("Terminal SoC drift (p.u.)", _fmt(summary["max_terminal_soc_violation"])),
    ])
end

function _check_fixed_dispatch_grid_condition(summary; n_worst::Int=10)
    uncertainty_bounds = summary["uncertainty_bounds"]
    microgrid_params = summary["microgrid_params"]
    s_star = summary["s_star"]
    pg_plus_max = microgrid_params["PG_plus_MAX"]
    pg_minus_max = microgrid_params["PG_minus_MAX"]
    delta_t = microgrid_params["Δt"]

    PL_width = uncertainty_bounds["PL_ub"] .- uncertainty_bounds["PL_lb"]
    PR_width = uncertainty_bounds["PR_ub"] .- uncertainty_bounds["PR_lb"]
    required_width = PL_width .+ s_star .* PR_width
    allowed_width = pg_plus_max + pg_minus_max
    margin = allowed_width .- required_width
    violated = margin .< -VALIDATION_TOL
    n_violated = count(violated)
    worst_order = sortperm(margin)
    n_show = min(n_worst, length(worst_order))

    result = Dict(
        "required_width" => required_width,
        "allowed_width" => allowed_width,
        "margin" => margin,
        "violated_mask" => violated,
        "n_violated" => n_violated,
        "violation_rate" => mean(violated),
        "min_margin" => minimum(margin),
        "worst_time_index" => worst_order[1],
    )

    println()
    println("Fixed-dispatch grid feasibility condition")
    println(repeat("-", 88))
    println("Condition: (PL_ub - PL_lb) + s*(PR_ub - PR_lb) <= PG_plus_MAX + PG_minus_MAX")
    println("PG_plus_MAX = ", pg_plus_max, " kW | PG_minus_MAX = ", pg_minus_max, " kW | s_star = ", s_star)
    println("Violated time periods = ", n_violated, " / ", length(margin), " (", round(100 * mean(violated); digits=4), "%)")
    println("Minimum margin = ", minimum(margin), " kW")
    println()
    println(rpad("t index", 10), " | ", rpad("time [h]", 10), " | ", rpad("required width [kW]", 22), " | ", rpad("allowed width [kW]", 20), " | margin [kW]")
    println(repeat("-", 88))
    for idx in worst_order[1:n_show]
        println(
            rpad(string(idx), 10), " | ",
            rpad(string(round((idx - 1) * delta_t; digits=3)), 10), " | ",
            rpad(string(round(required_width[idx]; digits=4)), 22), " | ",
            rpad(string(round(allowed_width; digits=4)), 20), " | ",
            round(margin[idx]; digits=4),
        )
    end

    return result
end

function _fixed_dispatch_grid_envelope(summary; n_worst::Int=10)
    uncertainty_bounds = summary["uncertainty_bounds"]
    microgrid_params = summary["microgrid_params"]
    grid_size = microgrid_params["grid_size"]
    n_t = grid_size - 1
    s_star = summary["s_star"]
    pg_plus_max = microgrid_params["PG_plus_MAX"]
    pg_minus_max = microgrid_params["PG_minus_MAX"]
    delta_t = microgrid_params["Δt"]
    Pc1, Pd1, Pc2, Pd2 = _unpack_u_star(summary["u_star"], grid_size; two_batt=summary["two_batt"])

    fixed_batt_net = Pc1 .+ Pc2 .- Pd1 .- Pd2
    r_min = fixed_batt_net .+ uncertainty_bounds["PL_lb"] .- s_star .* uncertainty_bounds["PR_ub"]
    r_max = fixed_batt_net .+ uncertainty_bounds["PL_ub"] .- s_star .* uncertainty_bounds["PR_lb"]
    upper_violation = max.(r_max .- pg_plus_max, 0.0)
    lower_violation = max.(-pg_minus_max .- r_min, 0.0)
    violation = max.(upper_violation, lower_violation)
    worst_order = sortperm(violation; rev=true)
    n_show = min(n_worst, n_t)

    result = Dict(
        "r_min" => r_min,
        "r_max" => r_max,
        "upper_violation" => upper_violation,
        "lower_violation" => lower_violation,
        "violation" => violation,
        "n_violated" => count(violation .> VALIDATION_TOL),
        "max_violation" => maximum(violation),
        "worst_time_index" => worst_order[1],
    )

    println()
    println("Exact fixed-dispatch grid envelope")
    println(repeat("-", 88))
    println("Condition: -PG_minus_MAX <= Pc + PL - s*PR - Pd <= PG_plus_MAX for all PL, PR")
    println("Violated intervals = ", result["n_violated"], " / ", n_t)
    println("Maximum robust grid violation = ", round(result["max_violation"]; digits=6), " kW")
    println()
    println(rpad("t index", 10), " | ", rpad("time [h]", 10), " | ", rpad("R_min [kW]", 16), " | ", rpad("R_max [kW]", 16), " | violation [kW]")
    println(repeat("-", 88))
    for idx in worst_order[1:n_show]
        println(
            rpad(string(idx), 10), " | ",
            rpad(string(round((idx - 1) * delta_t; digits=3)), 10), " | ",
            rpad(string(round(r_min[idx]; digits=4)), 16), " | ",
            rpad(string(round(r_max[idx]; digits=4)), 16), " | ",
            round(violation[idx]; digits=4),
        )
    end

    return result
end

function _soc_endpoint_envelope(Pc, Pd, n_units, uncertainty_bounds, microgrid_params)
    n_t = length(Pc)
    delta_t = microgrid_params["Δt"]
    soc0 = microgrid_params["SoC0"]
    e_batt = microgrid_params["E_BATT"]
    capacity = n_units * e_batt
    capacity > 0 || error("Battery capacity is non-positive: n_units=$n_units, E_BATT=$e_batt.")

    e_min = Vector{Float64}(undef, n_t + 1)
    e_max = Vector{Float64}(undef, n_t + 1)
    e_min[1] = soc0 * capacity
    e_max[1] = soc0 * capacity
    for t in 1:n_t
        e_min[t + 1] = e_min[t] + (uncertainty_bounds["eta_ch_lb"][t] * Pc[t] - uncertainty_bounds["rho_dis_ub"][t] * Pd[t]) * delta_t
        e_max[t + 1] = e_max[t] + (uncertainty_bounds["eta_ch_ub"][t] * Pc[t] - uncertainty_bounds["rho_dis_lb"][t] * Pd[t]) * delta_t
    end

    soc_min_env = e_min ./ capacity
    soc_max_env = e_max ./ capacity
    lower_violation = max.(microgrid_params["SOC_MIN"] .- soc_min_env, 0.0)
    upper_violation = max.(soc_max_env .- microgrid_params["SOC_MAX"], 0.0)
    violation = max.(lower_violation, upper_violation)

    return Dict(
        "capacity" => capacity,
        "soc_min" => soc_min_env,
        "soc_max" => soc_max_env,
        "lower_violation" => lower_violation,
        "upper_violation" => upper_violation,
        "violation" => violation,
        "max_violation" => maximum(violation),
        "terminal_violation" => max(abs(soc_min_env[end] - soc0), abs(soc_max_env[end] - soc0)),
    )
end

function _print_soc_endpoint_report(label, envelope, microgrid_params; is_cycle::Bool, n_worst::Int=10)
    delta_t = microgrid_params["Δt"]
    worst_order = sortperm(envelope["violation"]; rev=true)
    n_show = min(n_worst, length(worst_order))

    println()
    println("Exact fixed-dispatch SoC envelope: ", label)
    println(repeat("-", 88))
    println("Capacity = ", round(envelope["capacity"]; digits=6), " kWh")
    println("Violated state points = ", count(envelope["violation"] .> VALIDATION_TOL), " / ", length(envelope["violation"]))
    println("Maximum robust SoC violation = ", round(envelope["max_violation"]; digits=6), " p.u.")
    if is_cycle
        println("Robust terminal-cycle violation = ", round(envelope["terminal_violation"]; digits=6), " p.u.")
        println("Terminal SoC interval = [", round(envelope["soc_min"][end]; digits=6), ", ", round(envelope["soc_max"][end]; digits=6), "]")
    end
    println()
    println(rpad("x index", 10), " | ", rpad("time [h]", 10), " | ", rpad("SoC_min", 16), " | ", rpad("SoC_max", 16), " | violation")
    println(repeat("-", 88))
    for idx in worst_order[1:n_show]
        println(
            rpad(string(idx), 10), " | ",
            rpad(string(round((idx - 1) * delta_t; digits=3)), 10), " | ",
            rpad(string(round(envelope["soc_min"][idx]; digits=6)), 16), " | ",
            rpad(string(round(envelope["soc_max"][idx]; digits=6)), 16), " | ",
            round(envelope["violation"][idx]; digits=6),
        )
    end
end

function _check_fixed_dispatch_soc_condition(summary; n_worst::Int=10)
    uncertainty_bounds = summary["uncertainty_bounds"]
    microgrid_params = summary["microgrid_params"]
    grid_size = microgrid_params["grid_size"]
    Pc1, Pd1, Pc2, Pd2 = _unpack_u_star(summary["u_star"], grid_size; two_batt=summary["two_batt"])

    batt1 = _soc_endpoint_envelope(Pc1, Pd1, summary["n_star"], uncertainty_bounds, microgrid_params)
    batt2 = summary["two_batt"] ? _soc_endpoint_envelope(Pc2, Pd2, summary["m_star"], uncertainty_bounds, microgrid_params) : nothing

    _print_soc_endpoint_report("battery 1", batt1, microgrid_params; is_cycle=summary["is_cycle"], n_worst=n_worst)
    if summary["two_batt"]
        _print_soc_endpoint_report("battery 2", batt2, microgrid_params; is_cycle=summary["is_cycle"], n_worst=n_worst)
    end

    return Dict("battery1" => batt1, "battery2" => batt2)
end

function _plot_worst_scenario_trajectory(summary; case_key="worst_physical_case")
    _ensure_pyplot!() || error("PyPlot is not available. Install PyPlot or use the printed validation tables and deterministic checks.")
    haskey(summary, case_key) || error("Summary does not contain case `$case_key`.")
    case = summary[case_key]
    microgrid_params = summary["microgrid_params"]
    uncertainty_bounds = summary["uncertainty_bounds"]

    n_t = length(case["PL"])
    t_u = collect(0:n_t-1) .* microgrid_params["Δt"]
    t_x = collect(0:n_t) .* microgrid_params["Δt"]
    soc_min = microgrid_params["SOC_MIN"]
    soc_max = microgrid_params["SOC_MAX"]
    soc0 = microgrid_params["SoC0"]
    pg_plus_max = microgrid_params["PG_plus_MAX"]
    pg_minus_max = microgrid_params["PG_minus_MAX"]
    s_star = summary["s_star"]
    is_cycle = summary["is_cycle"]
    investment_cost = summary["investment_cost"]
    gamma_star = summary["gamma_star"]
    operating_cost_dt_factor = summary["operating_cost_dt_factor"]

    operating_cost_cumsum = operating_cost_dt_factor .* cumsum(case["C_buy"] .* case["PG_plus"] .- case["C_sell"] .* case["PG_minus"])
    epigraph_cumsum = investment_cost .+ operating_cost_cumsum

    fig, axs = plt.subplots(6, 1, figsize=(12, 14), sharex=false, constrained_layout=true)
    fig.suptitle("Monte Carlo worst scenario: " * replace(case_key, "_" => " "), fontsize=14)

    axs[1].plot(t_x, case["soc1"], color="black", linewidth=2, label="SoC pack 1")
    if summary["two_batt"]
        axs[1].plot(t_x, case["soc2"], color="tab:purple", linewidth=2, label="SoC pack 2")
    end
    axs[1].axhline(soc_min, color="red", linestyle="--", linewidth=1.5, label="SoC min/max")
    axs[1].axhline(soc_max, color="red", linestyle="--", linewidth=1.5)
    if is_cycle
        axs[1].axhline(soc0, color="tab:green", linestyle=":", linewidth=1.5, label="terminal target")
        axs[1].scatter([t_x[end]], [case["soc1"][end]], color="black", s=25)
        if summary["two_batt"]
            axs[1].scatter([t_x[end]], [case["soc2"][end]], color="tab:purple", s=25)
        end
    end
    axs[1].set_ylabel("SoC")
    axs[1].grid(true, alpha=0.3)
    axs[1].legend(loc="best")

    axs[2].plot(t_u, case["required_grid_net"], color="tab:blue", linewidth=2, label="required grid net")
    axs[2].axhline(pg_plus_max, color="red", linestyle="--", linewidth=1.5, label="import/export bounds")
    axs[2].axhline(-pg_minus_max, color="red", linestyle="--", linewidth=1.5)
    axs[2].fill_between(t_u, pg_plus_max, case["required_grid_net"], where=case["required_grid_net"] .> pg_plus_max, color="red", alpha=0.25)
    axs[2].fill_between(t_u, -pg_minus_max, case["required_grid_net"], where=case["required_grid_net"] .< -pg_minus_max, color="red", alpha=0.25)
    axs[2].set_ylabel("Grid net power")
    axs[2].grid(true, alpha=0.3)
    axs[2].legend(loc="best")

    axs[3].fill_between(t_u, uncertainty_bounds["PL_lb"], uncertainty_bounds["PL_ub"], color="tab:red", alpha=0.18, label="load lb/ub")
    axs[3].plot(t_u, case["PL"], color="tab:red", linewidth=1.8, label="load scenario")
    axs[3].fill_between(t_u, s_star .* uncertainty_bounds["PR_lb"], s_star .* uncertainty_bounds["PR_ub"], color="tab:orange", alpha=0.20, label="s* PV lb/ub")
    axs[3].plot(t_u, s_star .* case["PR"], color="tab:orange", linewidth=1.8, label="s* PV scenario")
    axs[3].set_ylabel("Power")
    axs[3].grid(true, alpha=0.3)
    axs[3].legend(loc="best")

    axs[4].fill_between(t_u, uncertainty_bounds["eta_ch_lb"], uncertainty_bounds["eta_ch_ub"], color="tab:blue", alpha=0.18, label="eta_ch lb/ub")
    axs[4].plot(t_u, case["eta_ch"], color="tab:blue", linewidth=1.8, label="eta_ch scenario")
    axs[4].fill_between(t_u, uncertainty_bounds["rho_dis_lb"], uncertainty_bounds["rho_dis_ub"], color="tab:purple", alpha=0.18, label="rho_dis lb/ub")
    axs[4].plot(t_u, case["rho_dis"], color="tab:purple", linewidth=1.8, label="rho_dis scenario")
    axs[4].set_ylabel("Efficiency")
    axs[4].grid(true, alpha=0.3)
    axs[4].legend(loc="best")

    axs[5].plot(t_u, case["PG_plus"], color="tab:green", linewidth=1.6, label="PG+")
    axs[5].plot(t_u, case["PG_minus"], color="tab:cyan", linewidth=1.6, label="PG-")
    axs[5].axhline(pg_plus_max, color="tab:green", linestyle="--", linewidth=1.2, label="PG+ max")
    axs[5].axhline(pg_minus_max, color="tab:cyan", linestyle="--", linewidth=1.2, label="PG- max")
    axs[5].set_ylabel("Grid power")
    axs[5].grid(true, alpha=0.3)
    axs[5].legend(loc="best")

    axs[6].plot(t_u, epigraph_cumsum, color="tab:blue", linewidth=2, label="investment + cumulative operating cost")
    axs[6].axhline(gamma_star, color="red", linestyle="--", linewidth=1.5, label="gamma*")
    axs[6].fill_between(t_u, gamma_star, epigraph_cumsum, where=epigraph_cumsum .> gamma_star, color="red", alpha=0.25)
    axs[6].set_ylabel("Cost")
    axs[6].set_xlabel("Time [h]")
    axs[6].grid(true, alpha=0.3)
    axs[6].legend(loc="best")

    println()
    println("Worst scenario plotted: ", case_key)
    println("  SoC bound violation (p.u.)      = ", case["soc_bound_violation"])
    println("  Terminal SoC drift (p.u.)       = ", case["terminal_soc_violation"])
    println("  Grid bound violation (kW)       = ", case["load_balance_violation"])
    println("  Epigraph violation (cost units) = ", case["epigraph_violation"])

    display(fig)
    return fig, axs
end

function _worst_case_dict(idx, PL, PR, C_buy, C_sell, eta_ch, rho_dis, soc1, soc2, PG_plus, PG_minus, required_grid_net, epigraph_values, epigraph_violations, soc_bound_violations, terminal_soc_violations, load_balance_violations, is_corner_scenario, corner_labels)
    return Dict(
        "PL" => PL[:, idx],
        "PR" => PR[:, idx],
        "C_buy" => C_buy[:, idx],
        "C_sell" => C_sell[:, idx],
        "eta_ch" => eta_ch[:, idx],
        "rho_dis" => rho_dis[:, idx],
        "soc1" => soc1[:, idx],
        "soc2" => soc2[:, idx],
        "PG_plus" => PG_plus[:, idx],
        "PG_minus" => PG_minus[:, idx],
        "required_grid_net" => required_grid_net[:, idx],
        "epigraph_value" => epigraph_values[idx],
        "epigraph_violation" => epigraph_violations[idx],
        "soc_bound_violation" => soc_bound_violations[idx],
        "terminal_soc_violation" => terminal_soc_violations[idx],
        "load_balance_violation" => load_balance_violations[idx],
        "is_corner_scenario" => is_corner_scenario[idx],
        "corner_label" => corner_labels[idx],
    )
end

function run_monte_carlo_from_result(result_path::AbstractString; n_scenarios::Int=100_000, solution_idx=nothing, operating_cost_dt_factor=nothing)
    result_data = JLD2.load(result_path)

    if haskey(result_data, "uncertainty_bounds") && haskey(result_data, "microgrid_params")
        uncertainty_bounds = result_data["uncertainty_bounds"]
        microgrid_params = result_data["microgrid_params"]
        resolved_data_path = get(result_data, "resolved_data_path", "")
    else
        resolved_data_path = get(result_data, "resolved_data_path", "")
        _ensure_problem_data_loader!()
        if isempty(resolved_data_path)
            _, uncertainty_bounds, _, _, _, _, microgrid_params = read_microgrid_problem_data()
        else
            _, uncertainty_bounds, _, _, _, _, microgrid_params = read_microgrid_problem_data(resolved_data_path)
        end
    end

    if operating_cost_dt_factor === nothing
        operating_cost_dt_factor = Float64(microgrid_params["Δt"])
    else
        operating_cost_dt_factor = Float64(operating_cost_dt_factor)
        operating_cost_dt_factor >= 0 || error("operating_cost_dt_factor must be nonnegative.")
    end

    u_star, gamma_star, selected_solution_idx = _extract_solution_from_result(result_data; solution_idx=solution_idx)
    last_solution_idx = _last_finite_col(result_data["u_star_history"])
    if solution_idx !== nothing && selected_solution_idx != last_solution_idx
        @warn "Replaying solution index $selected_solution_idx, but the last finite saved solution is $last_solution_idx."
    end
    solve_stats = haskey(result_data, "solve_stats") ? result_data["solve_stats"] : nothing
    formulation = haskey(result_data, "is_milp") && Bool(result_data["is_milp"]) ? "MILP" : "NLP"
    optimizer_solve_count = Int(_stats_value(solve_stats, :nlp_count; default=0)) + Int(_stats_value(solve_stats, :lp_count; default=0))
    local_reduction_runtime_seconds = Float64(_result_field(result_data, "total_elapsed_seconds"; default=_stats_value(solve_stats, :local_red_time; default=NaN)))
    n_star = _result_value(result_data, "n_star")
    m_star = _result_value(result_data, "m_star"; default=0.0)
    s_star = _result_value(result_data, "s_star")
    isfinite(n_star) || error("Result file does not contain finite `n_star`.")
    isfinite(s_star) || error("Result file does not contain finite `s_star`.")

    grid_size = microgrid_params["grid_size"]
    n_t = grid_size - 1
    two_batt = haskey(result_data, "two_batt") ? Bool(result_data["two_batt"]) : div(length(u_star), n_t) == 4
    is_cycle = haskey(result_data, "is_cycle") ? Bool(result_data["is_cycle"]) : false
    Pc1, Pd1, Pc2, Pd2 = _unpack_u_star(u_star, grid_size; two_batt=two_batt)

    random_scenarios = _sample_uniform_scenarios(uncertainty_bounds, n_scenarios, grid_size)
    corner_scenarios = _global_corner_scenarios(uncertainty_bounds, grid_size)
    PL = hcat(random_scenarios["PL"], corner_scenarios["PL"])
    PR = hcat(random_scenarios["PR"], corner_scenarios["PR"])
    C_buy = hcat(random_scenarios["C_buy"], corner_scenarios["C_buy"])
    C_sell = hcat(random_scenarios["C_sell"], corner_scenarios["C_sell"])
    eta_ch = hcat(random_scenarios["eta_ch"], corner_scenarios["eta_ch"])
    rho_dis = hcat(random_scenarios["rho_dis"], corner_scenarios["rho_dis"])
    total_scenarios = size(PL, 2)
    is_corner_scenario = vcat(falses(n_scenarios), corner_scenarios["corner_flags"])
    corner_labels = vcat(fill("", n_scenarios), corner_scenarios["corner_labels"])

    Pc1_mat = repeat(reshape(Pc1, n_t, 1), 1, total_scenarios)
    Pd1_mat = repeat(reshape(Pd1, n_t, 1), 1, total_scenarios)
    Pc2_mat = repeat(reshape(Pc2, n_t, 1), 1, total_scenarios)
    Pd2_mat = repeat(reshape(Pd2, n_t, 1), 1, total_scenarios)

    required_grid_net = Pc1_mat .+ Pc2_mat .+ PL .- s_star .* PR .- Pd1_mat .- Pd2_mat
    PG_plus = max.(required_grid_net, 0.0)
    PG_minus = max.(-required_grid_net, 0.0)
    pg_plus_max = microgrid_params["PG_plus_MAX"]
    pg_minus_max = microgrid_params["PG_minus_MAX"]
    # This is a grid-capacity violation. Load balance itself is enforced by
    # reconstructing PG+ and PG- from required_grid_net.
    load_balance_violation_t = max.(required_grid_net .- pg_plus_max, -pg_minus_max .- required_grid_net, 0.0)
    load_balance_violation_by_scenario = _scenario_max(load_balance_violation_t)
    load_balance_violated = load_balance_violation_by_scenario .> VALIDATION_TOL

    soc1, soc2, energy1, energy2 = _propagate_soc(Pc1_mat, Pd1_mat, Pc2_mat, Pd2_mat, eta_ch, rho_dis, microgrid_params, n_star, m_star; two_batt=two_batt)
    soc_min = microgrid_params["SOC_MIN"]
    soc_max = microgrid_params["SOC_MAX"]
    soc1_bound_violation_t = max.(soc_min .- soc1, soc1 .- soc_max, 0.0)
    soc2_bound_violation_t = two_batt ? max.(soc_min .- soc2, soc2 .- soc_max, 0.0) : zeros(size(soc1))
    soc_bound_violation_by_scenario = _scenario_max(max.(soc1_bound_violation_t, soc2_bound_violation_t))
    soc_bound_violated = soc_bound_violation_by_scenario .> VALIDATION_TOL
    soc_bound_violation_counts = _scenario_count(soc1_bound_violation_t .> VALIDATION_TOL)
    soc_bound_violation_counts .+= two_batt ? _scenario_count(soc2_bound_violation_t .> VALIDATION_TOL) : zeros(total_scenarios)
    soc_bound_constraint_count = (two_batt ? 2 : 1) * (n_t + 1)

    soc0 = microgrid_params["SoC0"]
    terminal_soc1_violation = abs.(soc1[end, :] .- soc0)
    terminal_soc2_violation = two_batt ? abs.(soc2[end, :] .- soc0) : zeros(total_scenarios)
    terminal_soc_violation_by_scenario = is_cycle ? max.(terminal_soc1_violation, terminal_soc2_violation) : zeros(total_scenarios)
    terminal_soc_violated = terminal_soc_violation_by_scenario .> VALIDATION_TOL
    terminal_soc_violation_counts = is_cycle ? Float64.(terminal_soc1_violation .> VALIDATION_TOL) : zeros(total_scenarios)
    terminal_soc_violation_counts .+= (is_cycle && two_batt) ? Float64.(terminal_soc2_violation .> VALIDATION_TOL) : zeros(total_scenarios)

    operating_costs_unscaled = vec(sum(C_buy .* PG_plus .- C_sell .* PG_minus; dims=1))
    operating_costs = operating_cost_dt_factor .* operating_costs_unscaled
    e_batt = microgrid_params["E_BATT"]
    battery_cost = _param_value(result_data, microgrid_params, "battery_cost"; aliases=["Battery_Cost"], default=200.0)
    pv_cost = _param_value(result_data, microgrid_params, "pv_cost"; aliases=["PV_Cost"], default=1500.0)
    base_pv_capacity_kw = _param_value(result_data, microgrid_params, "base_pv_capacity_kw"; aliases=["Base_PV_Capacity_kW"], default=90.84)
    investment_cost = (n_star + (two_batt ? m_star : 0.0)) * e_batt * battery_cost + s_star * base_pv_capacity_kw * pv_cost
    epigraph_values = investment_cost .+ operating_costs
    epigraph_violations = epigraph_values .- gamma_star
    epigraph_violated = epigraph_violations .> VALIDATION_TOL
    epigraph_violation_counts = Float64.(epigraph_violated)
    epigraph_constraint_count = 1

    load_balance_violation_counts = _scenario_count(load_balance_violation_t .> VALIDATION_TOL)
    load_balance_constraint_count = n_t
    terminal_soc_constraint_count = is_cycle ? (two_batt ? 2 : 1) : 0
    total_constraint_count = soc_bound_constraint_count + load_balance_constraint_count + terminal_soc_constraint_count + epigraph_constraint_count
    violated_constraint_counts = soc_bound_violation_counts .+ load_balance_violation_counts .+ terminal_soc_violation_counts .+ epigraph_violation_counts
    constraint_violation_pct_by_scenario = _constraint_percent(violated_constraint_counts, total_constraint_count)
    soc_bound_constraint_violation_pct_by_scenario = _constraint_percent(soc_bound_violation_counts, soc_bound_constraint_count)
    load_balance_constraint_violation_pct_by_scenario = _constraint_percent(load_balance_violation_counts, load_balance_constraint_count)
    terminal_soc_constraint_violation_pct_by_scenario = _constraint_percent(terminal_soc_violation_counts, terminal_soc_constraint_count)
    epigraph_constraint_violation_pct_by_scenario = _constraint_percent(epigraph_violation_counts, epigraph_constraint_count)
    max_overall_violation = maximum(max.(
        soc_bound_violation_by_scenario,
        load_balance_violation_by_scenario,
        terminal_soc_violation_by_scenario,
        epigraph_violations,
        0.0,
    ))

    physical_infeasible_mask = soc_bound_violated .| load_balance_violated .| terminal_soc_violated
    infeasible_mask = physical_infeasible_mask .| epigraph_violated
    feasible_rate = 1.0 - mean(infeasible_mask)
    physical_feasible_rate = 1.0 - mean(physical_infeasible_mask)
    n_infeasible = count(infeasible_mask)
    n_physical_infeasible = count(physical_infeasible_mask)
    infeasible_scenario_soc_bound_share = n_infeasible == 0 ? 0.0 : count(infeasible_mask .& soc_bound_violated) / n_infeasible
    infeasible_scenario_load_balance_share = n_infeasible == 0 ? 0.0 : count(infeasible_mask .& load_balance_violated) / n_infeasible
    infeasible_scenario_terminal_soc_share = n_infeasible == 0 ? 0.0 : count(infeasible_mask .& terminal_soc_violated) / n_infeasible
    infeasible_scenario_epigraph_share = n_infeasible == 0 ? 0.0 : count(infeasible_mask .& epigraph_violated) / n_infeasible
    infeasible_scenario_soc_and_load_share = n_infeasible == 0 ? 0.0 : count(infeasible_mask .& soc_bound_violated .& load_balance_violated) / n_infeasible

    worst_epigraph_idx = argmax(epigraph_violations)
    worst_soc_idx = argmax(soc_bound_violation_by_scenario)
    worst_terminal_idx = argmax(terminal_soc_violation_by_scenario)
    worst_balance_idx = argmax(load_balance_violation_by_scenario)
    physical_violation_by_scenario = max.(soc_bound_violation_by_scenario, load_balance_violation_by_scenario, terminal_soc_violation_by_scenario)
    worst_physical_idx = argmax(physical_violation_by_scenario)
    infeasible_indices = findall(infeasible_mask)
    infeasible_corner_indices = [idx for idx in infeasible_indices if is_corner_scenario[idx]]
    infeasible_random_indices = [idx for idx in infeasible_indices if !is_corner_scenario[idx]]

    summary = Dict(
        "result_path" => String(result_path),
        "resolved_data_path" => String(resolved_data_path),
        "solution_index" => selected_solution_idx,
        "last_solution_index" => last_solution_idx,
        "n_scenarios_random" => n_scenarios,
        "n_corner_scenarios" => corner_scenarios["n_corners"],
        "n_scenarios" => total_scenarios,
        "formulation" => formulation,
        "local_reduction_runtime_seconds" => local_reduction_runtime_seconds,
        "optimizer_solve_count" => optimizer_solve_count,
        "two_batt" => two_batt,
        "is_cycle" => is_cycle,
        "n_star" => n_star,
        "m_star" => m_star,
        "s_star" => s_star,
        "gamma_star" => gamma_star,
        "operating_cost_dt_factor" => Float64(operating_cost_dt_factor),
        "microgrid_params" => microgrid_params,
        "uncertainty_bounds" => uncertainty_bounds,
        "feasible_rate" => feasible_rate,
        "infeasible_rate" => mean(infeasible_mask),
        "n_infeasible" => n_infeasible,
        "physical_feasible_rate" => physical_feasible_rate,
        "physical_infeasible_rate" => mean(physical_infeasible_mask),
        "n_physical_infeasible" => n_physical_infeasible,
        "infeasible_scenario_soc_bound_share" => infeasible_scenario_soc_bound_share,
        "infeasible_scenario_load_balance_share" => infeasible_scenario_load_balance_share,
        "infeasible_scenario_terminal_soc_share" => infeasible_scenario_terminal_soc_share,
        "infeasible_scenario_epigraph_share" => infeasible_scenario_epigraph_share,
        "infeasible_scenario_soc_and_load_share" => infeasible_scenario_soc_and_load_share,
        "soc_bound_violation_rate" => mean(soc_bound_violated),
        "terminal_soc_violation_rate" => mean(terminal_soc_violated),
        "load_balance_violation_rate" => mean(load_balance_violated),
        "epigraph_violation_rate" => mean(epigraph_violated),
        "max_soc_bound_violation" => maximum(soc_bound_violation_by_scenario),
        "max_terminal_soc_violation" => maximum(terminal_soc_violation_by_scenario),
        "max_load_balance_violation" => maximum(load_balance_violation_by_scenario),
        "max_epigraph_violation" => maximum(epigraph_violations),
        "max_overall_violation" => max_overall_violation,
        "total_constraint_count" => total_constraint_count,
        "soc_bound_constraint_count" => soc_bound_constraint_count,
        "load_balance_constraint_count" => load_balance_constraint_count,
        "terminal_soc_constraint_count" => terminal_soc_constraint_count,
        "epigraph_constraint_count" => epigraph_constraint_count,
        "max_constraint_violation_pct" => maximum(constraint_violation_pct_by_scenario),
        "avg_constraint_violation_pct" => mean(constraint_violation_pct_by_scenario),
        "avg_constraint_violation_pct_infeasible" => _mean_on_mask(constraint_violation_pct_by_scenario, infeasible_mask),
        "avg_soc_bound_constraint_violation_pct_infeasible" => _mean_on_mask(soc_bound_constraint_violation_pct_by_scenario, infeasible_mask),
        "avg_load_balance_constraint_violation_pct_infeasible" => _mean_on_mask(load_balance_constraint_violation_pct_by_scenario, infeasible_mask),
        "avg_terminal_soc_constraint_violation_pct_infeasible" => _mean_on_mask(terminal_soc_constraint_violation_pct_by_scenario, infeasible_mask),
        "avg_epigraph_constraint_violation_pct_infeasible" => _mean_on_mask(epigraph_constraint_violation_pct_by_scenario, infeasible_mask),
        "investment_cost" => investment_cost,
        "soc_bound_violations" => soc_bound_violation_by_scenario,
        "terminal_soc_violations" => terminal_soc_violation_by_scenario,
        "load_balance_violations" => load_balance_violation_by_scenario,
        "grid_bound_violations" => load_balance_violation_by_scenario,
        "physical_violations" => physical_violation_by_scenario,
        "epigraph_values" => epigraph_values,
        "epigraph_violations" => epigraph_violations,
        "operating_costs_unscaled" => operating_costs_unscaled,
        "operating_costs" => operating_costs,
        "violated_constraint_counts" => violated_constraint_counts,
        "constraint_violation_pct_by_scenario" => constraint_violation_pct_by_scenario,
        "soc_bound_constraint_violation_pct_by_scenario" => soc_bound_constraint_violation_pct_by_scenario,
        "load_balance_constraint_violation_pct_by_scenario" => load_balance_constraint_violation_pct_by_scenario,
        "terminal_soc_constraint_violation_pct_by_scenario" => terminal_soc_constraint_violation_pct_by_scenario,
        "epigraph_constraint_violation_pct_by_scenario" => epigraph_constraint_violation_pct_by_scenario,
        "infeasible_mask" => infeasible_mask,
        "physical_infeasible_mask" => physical_infeasible_mask,
        "soc_bound_violated_mask" => soc_bound_violated,
        "terminal_soc_violated_mask" => terminal_soc_violated,
        "load_balance_violated_mask" => load_balance_violated,
        "grid_bound_violated_mask" => load_balance_violated,
        "epigraph_violated_mask" => epigraph_violated,
        "is_corner_scenario" => is_corner_scenario,
        "corner_labels" => corner_labels,
        "infeasible_indices" => infeasible_indices,
        "infeasible_corner_indices" => infeasible_corner_indices,
        "infeasible_random_indices" => infeasible_random_indices,
        "worst_physical_case" => _worst_case_dict(worst_physical_idx, PL, PR, C_buy, C_sell, eta_ch, rho_dis, soc1, soc2, PG_plus, PG_minus, required_grid_net, epigraph_values, epigraph_violations, soc_bound_violation_by_scenario, terminal_soc_violation_by_scenario, load_balance_violation_by_scenario, is_corner_scenario, corner_labels),
        "worst_epigraph_case" => _worst_case_dict(worst_epigraph_idx, PL, PR, C_buy, C_sell, eta_ch, rho_dis, soc1, soc2, PG_plus, PG_minus, required_grid_net, epigraph_values, epigraph_violations, soc_bound_violation_by_scenario, terminal_soc_violation_by_scenario, load_balance_violation_by_scenario, is_corner_scenario, corner_labels),
        "worst_soc_bound_case" => _worst_case_dict(worst_soc_idx, PL, PR, C_buy, C_sell, eta_ch, rho_dis, soc1, soc2, PG_plus, PG_minus, required_grid_net, epigraph_values, epigraph_violations, soc_bound_violation_by_scenario, terminal_soc_violation_by_scenario, load_balance_violation_by_scenario, is_corner_scenario, corner_labels),
        "worst_terminal_soc_case" => _worst_case_dict(worst_terminal_idx, PL, PR, C_buy, C_sell, eta_ch, rho_dis, soc1, soc2, PG_plus, PG_minus, required_grid_net, epigraph_values, epigraph_violations, soc_bound_violation_by_scenario, terminal_soc_violation_by_scenario, load_balance_violation_by_scenario, is_corner_scenario, corner_labels),
        "worst_load_balance_case" => _worst_case_dict(worst_balance_idx, PL, PR, C_buy, C_sell, eta_ch, rho_dis, soc1, soc2, PG_plus, PG_minus, required_grid_net, epigraph_values, epigraph_violations, soc_bound_violation_by_scenario, terminal_soc_violation_by_scenario, load_balance_violation_by_scenario, is_corner_scenario, corner_labels),
        "u_star" => u_star,
        "soc1" => soc1,
        "soc2" => soc2,
        "energy1" => energy1,
        "energy2" => energy2,
        "PG_plus" => PG_plus,
        "PG_minus" => PG_minus,
        "required_grid_net" => required_grid_net,
    )

    _print_mc_results_tables(summary)
    println()
    println("Scenario details")
    println(repeat("-", 88))
    println(rpad("Result file", 30), " | ", result_path)
    println(rpad("Scenarios", 30), " | ", total_scenarios, " (", n_scenarios, " random + ", corner_scenarios["n_corners"], " corners)")
    println(rpad("n_star, m_star, s_star", 30), " | ", n_star, ", ", m_star, ", ", s_star)
    println(rpad("Constraint count", 30), " | ", total_constraint_count, " per scenario")
    println(rpad("Infeasible definition", 30), " | robust = SoC bounds OR grid bounds OR terminal cycle OR epigraph")
    println(rpad("Physical infeasible definition", 30), " | physical = SoC bounds OR grid bounds OR terminal cycle")
    if !isempty(infeasible_corner_indices)
        println(rpad("Robust-infeasible corners", 30), " | ", [corner_labels[idx] for idx in infeasible_corner_indices])
    else
        println(rpad("Robust-infeasible corners", 30), " | none")
    end

    return summary
end

function _parse_cli(args)
    run_result_path = isempty(args) ? result_path : args[1]
    run_solution_idx = selected_solution_idx
    if length(args) >= 2
        token = lowercase(args[2])
        run_solution_idx = token in ("last", "nothing", "final") ? nothing : parse(Int, args[2])
    end
    run_n_scenarios = length(args) >= 3 ? parse(Int, args[3]) : 100_000
    run_dt_factor = length(args) >= 4 ? parse(Float64, args[4]) : nothing
    return run_result_path, run_solution_idx, run_n_scenarios, run_dt_factor
end

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    cli_result_path, cli_solution_idx, cli_n_scenarios, cli_dt_factor = _parse_cli(ARGS)
    if !isfile(cli_result_path)
        @warn "Set `result_path` in MonteCarloFromResult.jl or pass one saved .jld2 file as the first argument before running." result_path=cli_result_path
        exit(1)
    end

    mc = run_monte_carlo_from_result(
        cli_result_path;
        n_scenarios=cli_n_scenarios,
        solution_idx=cli_solution_idx,
        operating_cost_dt_factor=cli_dt_factor,
    )
    grid_width_condition = _check_fixed_dispatch_grid_condition(mc)
    grid_envelope = _fixed_dispatch_grid_envelope(mc)
    soc_envelope = _check_fixed_dispatch_soc_condition(mc)
    if lowercase(get(ENV, "LR_PLOT_WORST", "0")) in ("1", "true", "yes") && _ensure_pyplot!()
        fig_worst, axs_worst = _plot_worst_scenario_trajectory(mc)
    end
end
