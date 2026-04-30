using JLD2
using Printf

# Usage:
#   julia VerifySavedResult.jl [result_path] [solution_idx|last] [rows_to_show]
#
# This script only needs JLD2. It reads all model data needed for the checks
# from the saved result file and does not rerun the optimizer or Monte Carlo.
const DEFAULT_RESULT_PATH = joinpath(@__DIR__, "Results", "MILP_SOC060_H1_PG100_B1_2026-04-28_123251.jld2")

function _last_finite_col(matrix)
    for j in size(matrix, 2):-1:1
        any(isfinite, matrix[:, j]) && return j
    end
    error("Could not find a finite solution column in `u_star_history`.")
end

function _result_float(result_data, key; default=nothing)
    if haskey(result_data, key)
        return Float64(result_data[key])
    end
    default === nothing && error("Result file does not contain `$key`.")
    return Float64(default)
end

function _param_float(result_data, microgrid_params, key; default=nothing)
    if haskey(result_data, key)
        return Float64(result_data[key])
    elseif haskey(microgrid_params, key)
        return Float64(microgrid_params[key])
    elseif haskey(result_data, "data_config") && hasproperty(result_data["data_config"], Symbol(key))
        return Float64(getproperty(result_data["data_config"], Symbol(key)))
    end
    default === nothing && error("Could not find parameter `$key`.")
    return Float64(default)
end

function _extract_solution(result_data; solution_idx=nothing)
    haskey(result_data, "u_star_history") || error("Result file does not contain `u_star_history`.")
    haskey(result_data, "master_x_obj_history") || error("Result file does not contain `master_x_obj_history`.")

    u_star_history = result_data["u_star_history"]
    obj_history = result_data["master_x_obj_history"]
    idx = solution_idx === nothing ? _last_finite_col(u_star_history) : Int(solution_idx)
    1 <= idx <= size(u_star_history, 2) || error("solution_idx=$idx is outside 1:$(size(u_star_history, 2)).")
    any(isfinite, u_star_history[:, idx]) || error("u_star_history column $idx is not finite.")
    isfinite(obj_history[idx]) || error("master_x_obj_history[$idx] is not finite.")

    return u_star_history[:, idx], Float64(obj_history[idx]), idx
end

function _unpack_dispatch(u_star, n_t; two_batt::Bool)
    u_dim = div(length(u_star), n_t)
    length(u_star) == u_dim * n_t || error("u_star length $(length(u_star)) is incompatible with n_t=$n_t.")
    expected = two_batt ? 4 : 2
    u_dim == expected || error("Expected $expected controls per time step, got $u_dim.")

    pc1 = [u_star[u_dim * (t - 1) + 1] for t in 1:n_t]
    pd1 = [u_star[u_dim * (t - 1) + 2] for t in 1:n_t]
    pc2 = two_batt ? [u_star[u_dim * (t - 1) + 3] for t in 1:n_t] : zeros(n_t)
    pd2 = two_batt ? [u_star[u_dim * (t - 1) + 4] for t in 1:n_t] : zeros(n_t)
    return pc1, pd1, pc2, pd2
end

function _top_indices(values; n_show::Int=8, largest::Bool=true)
    n = min(n_show, length(values))
    if largest
        return partialsortperm(values, 1:n; rev=true)
    end
    return partialsortperm(values, 1:n)
end

function _print_section(title)
    println()
    println(title)
    println(repeat("-", 96))
end

function _fmt(x; digits::Int=6)
    if !isfinite(x)
        return string(x)
    end
    return @sprintf("%.*f", digits, x)
end

function _print_kv(label, value)
    println(rpad(label, 38), " : ", value)
end

function _time_hours(idx, delta_t; state_grid::Bool=false)
    return (idx - 1) * delta_t
end

function _grid_diagnostics(pc1, pd1, pc2, pd2, s_star, uncertainty_bounds, microgrid_params)
    pl_lb = Float64.(uncertainty_bounds["PL_lb"])
    pl_ub = Float64.(uncertainty_bounds["PL_ub"])
    pr_lb = Float64.(uncertainty_bounds["PR_lb"])
    pr_ub = Float64.(uncertainty_bounds["PR_ub"])
    pg_plus_max = Float64(microgrid_params["PG_plus_MAX"])
    pg_minus_max = Float64(microgrid_params["PG_minus_MAX"])

    fixed_batt_net = pc1 .+ pc2 .- pd1 .- pd2
    r_min = fixed_batt_net .+ pl_lb .- s_star .* pr_ub
    r_max = fixed_batt_net .+ pl_ub .- s_star .* pr_lb
    upper_violation = max.(r_max .- pg_plus_max, 0.0)
    lower_violation = max.(-pg_minus_max .- r_min, 0.0)
    violation = max.(upper_violation, lower_violation)
    required_width = (pl_ub .- pl_lb) .+ s_star .* (pr_ub .- pr_lb)
    allowed_width = pg_plus_max + pg_minus_max
    width_margin = allowed_width .- required_width

    return (
        fixed_batt_net=fixed_batt_net,
        r_min=r_min,
        r_max=r_max,
        upper_violation=upper_violation,
        lower_violation=lower_violation,
        violation=violation,
        required_width=required_width,
        allowed_width=allowed_width,
        width_margin=width_margin,
    )
end

function _soc_envelope(pc, pd, n_units, uncertainty_bounds, microgrid_params)
    n_t = length(pc)
    eta_ch_lb = Float64.(uncertainty_bounds["eta_ch_lb"])
    eta_ch_ub = Float64.(uncertainty_bounds["eta_ch_ub"])
    rho_dis_lb = haskey(uncertainty_bounds, "rho_dis_lb") ? Float64.(uncertainty_bounds["rho_dis_lb"]) : 1.0 ./ Float64.(uncertainty_bounds["eta_dis_ub"])
    rho_dis_ub = haskey(uncertainty_bounds, "rho_dis_ub") ? Float64.(uncertainty_bounds["rho_dis_ub"]) : 1.0 ./ Float64.(uncertainty_bounds["eta_dis_lb"])

    delta_t = Float64(microgrid_params["Δt"])
    soc0 = Float64(microgrid_params["SoC0"])
    e_batt = Float64(microgrid_params["E_BATT"])
    capacity = n_units * e_batt
    capacity > 0 || error("Battery capacity is non-positive: n_units=$n_units, E_BATT=$e_batt.")

    e_min = Vector{Float64}(undef, n_t + 1)
    e_max = Vector{Float64}(undef, n_t + 1)
    e_min[1] = soc0 * capacity
    e_max[1] = soc0 * capacity
    for t in 1:n_t
        e_min[t + 1] = e_min[t] + (eta_ch_lb[t] * pc[t] - rho_dis_ub[t] * pd[t]) * delta_t
        e_max[t + 1] = e_max[t] + (eta_ch_ub[t] * pc[t] - rho_dis_lb[t] * pd[t]) * delta_t
    end

    soc_min_env = e_min ./ capacity
    soc_max_env = e_max ./ capacity
    soc_min_bound = Float64(microgrid_params["SOC_MIN"])
    soc_max_bound = Float64(microgrid_params["SOC_MAX"])
    lower_violation = max.(soc_min_bound .- soc_min_env, 0.0)
    upper_violation = max.(soc_max_env .- soc_max_bound, 0.0)
    violation = max.(lower_violation, upper_violation)

    return (
        capacity=capacity,
        e_min=e_min,
        e_max=e_max,
        soc_min=soc_min_env,
        soc_max=soc_max_env,
        lower_violation=lower_violation,
        upper_violation=upper_violation,
        violation=violation,
    )
end

function _control_diagnostics(pc1, pd1, pc2, pd2, n_star, m_star, microgrid_params; two_batt::Bool)
    pc_max = Float64(microgrid_params["Pc_MAX"])
    pd_max = Float64(microgrid_params["Pd_MAX"])
    pc1_violation = max.(pc1 .- n_star * pc_max, -pc1, 0.0)
    pd1_violation = max.(pd1 .- n_star * pd_max, -pd1, 0.0)
    if two_batt
        pc2_violation = max.(pc2 .- m_star * pc_max, -pc2, 0.0)
        pd2_violation = max.(pd2 .- m_star * pd_max, -pd2, 0.0)
    else
        pc2_violation = zeros(length(pc1))
        pd2_violation = zeros(length(pc1))
    end
    return max.(pc1_violation, pd1_violation, pc2_violation, pd2_violation)
end

function _max_price_cost_at_net(r, c_buy_ub, c_sell_lb)
    return r >= 0 ? c_buy_ub * r : c_sell_lb * r
end

function _epigraph_diagnostics(grid_diag, result_data, uncertainty_bounds, microgrid_params, n_star, m_star, s_star, gamma_star; two_batt::Bool)
    c_buy_ub = Float64.(uncertainty_bounds["C_buy_ub"])
    c_sell_lb = Float64.(uncertainty_bounds["C_sell_lb"])
    n_t = length(c_buy_ub)
    per_period_code = zeros(n_t)
    r_argmax = zeros(n_t)

    for t in 1:n_t
        candidates = [grid_diag.r_min[t], grid_diag.r_max[t]]
        if grid_diag.r_min[t] <= 0.0 <= grid_diag.r_max[t]
            push!(candidates, 0.0)
        end
        values = [_max_price_cost_at_net(r, c_buy_ub[t], c_sell_lb[t]) for r in candidates]
        best = argmax(values)
        per_period_code[t] = values[best]
        r_argmax[t] = candidates[best]
    end

    e_batt = Float64(microgrid_params["E_BATT"])
    battery_cost = _param_float(result_data, microgrid_params, "battery_cost"; default=200.0)
    pv_cost = _param_float(result_data, microgrid_params, "pv_cost"; default=1500.0)
    base_pv_capacity_kw = _param_float(result_data, microgrid_params, "base_pv_capacity_kw"; default=90.84)
    delta_t = Float64(microgrid_params["Δt"])
    investment = (n_star + (two_batt ? m_star : 0.0)) * e_batt * battery_cost + s_star * base_pv_capacity_kw * pv_cost
    operating_code = sum(per_period_code)
    operating_ifac = delta_t * operating_code

    return (
        battery_cost=battery_cost,
        pv_cost=pv_cost,
        base_pv_capacity_kw=base_pv_capacity_kw,
        investment=investment,
        per_period_code=per_period_code,
        r_argmax=r_argmax,
        operating_code=operating_code,
        operating_ifac=operating_ifac,
        total_code=investment + operating_code,
        total_ifac=investment + operating_ifac,
        violation_code=investment + operating_code - gamma_star,
        violation_ifac=investment + operating_ifac - gamma_star,
    )
end

function _print_grid_report(grid_diag, microgrid_params; n_show::Int=8)
    delta_t = Float64(microgrid_params["Δt"])
    _print_section("Exact Grid Feasibility Envelope")
    _print_kv("Max grid violation [kW]", _fmt(maximum(grid_diag.violation)))
    _print_kv("Violated intervals", "$(count(grid_diag.violation .> 1e-7)) / $(length(grid_diag.violation))")
    _print_kv("Worst interval-width margin [kW]", _fmt(minimum(grid_diag.width_margin)))
    _print_kv("Width condition violated", "$(count(grid_diag.width_margin .< -1e-7)) / $(length(grid_diag.width_margin))")

    println()
    println(rpad("t", 6), rpad("hour", 10), rpad("R_min", 16), rpad("R_max", 16), rpad("viol", 16), rpad("width margin", 16))
    println(repeat("-", 96))
    for idx in _top_indices(grid_diag.violation; n_show=n_show, largest=true)
        println(
            rpad(string(idx), 6),
            rpad(_fmt(_time_hours(idx, delta_t); digits=3), 10),
            rpad(_fmt(grid_diag.r_min[idx]), 16),
            rpad(_fmt(grid_diag.r_max[idx]), 16),
            rpad(_fmt(grid_diag.violation[idx]), 16),
            rpad(_fmt(grid_diag.width_margin[idx]), 16),
        )
    end
end

function _print_soc_report(label, soc_diag, microgrid_params; is_cycle::Bool, n_show::Int=8)
    delta_t = Float64(microgrid_params["Δt"])
    soc0 = Float64(microgrid_params["SoC0"])
    _print_section("Exact SoC Envelope: $label")
    _print_kv("Capacity [kWh]", _fmt(soc_diag.capacity))
    _print_kv("Max SoC bound violation [p.u.]", _fmt(maximum(soc_diag.violation)))
    _print_kv("Violated state points", "$(count(soc_diag.violation .> 1e-7)) / $(length(soc_diag.violation))")
    if is_cycle
        terminal_violation = max(abs(soc_diag.soc_min[end] - soc0), abs(soc_diag.soc_max[end] - soc0))
        _print_kv("Terminal equality violation [p.u.]", _fmt(terminal_violation))
        _print_kv("Terminal SoC interval", "[" * _fmt(soc_diag.soc_min[end]) * ", " * _fmt(soc_diag.soc_max[end]) * "]")
    end

    println()
    println(rpad("x index", 8), rpad("hour", 10), rpad("SoC_min_possible", 20), rpad("SoC_max_possible", 20), rpad("viol", 16))
    println(repeat("-", 96))
    for idx in _top_indices(soc_diag.violation; n_show=n_show, largest=true)
        println(
            rpad(string(idx), 8),
            rpad(_fmt(_time_hours(idx, delta_t; state_grid=true); digits=3), 10),
            rpad(_fmt(soc_diag.soc_min[idx]), 20),
            rpad(_fmt(soc_diag.soc_max[idx]), 20),
            rpad(_fmt(soc_diag.violation[idx]), 16),
        )
    end
end

function _print_control_report(control_violation)
    _print_section("Control Bound Check")
    _print_kv("Max control-bound violation [kW]", _fmt(maximum(control_violation)))
    _print_kv("Violated intervals", "$(count(control_violation .> 1e-7)) / $(length(control_violation))")
end

function _print_epigraph_report(epi_diag, gamma_star, microgrid_params; n_show::Int=8)
    delta_t = Float64(microgrid_params["Δt"])
    _print_section("Exact Epigraph Cost Envelope")
    _print_kv("Investment cost", _fmt(epi_diag.investment))
    _print_kv("gamma_star", _fmt(gamma_star))
    _print_kv("Worst operating cost, legacy unscaled", _fmt(epi_diag.operating_code))
    _print_kv("Worst total cost, legacy unscaled", _fmt(epi_diag.total_code))
    _print_kv("Worst epigraph violation, legacy", _fmt(epi_diag.violation_code))
    _print_kv("Worst operating cost, Delta t", _fmt(epi_diag.operating_ifac))
    _print_kv("Worst total cost, Delta t", _fmt(epi_diag.total_ifac))
    _print_kv("Worst epigraph violation, Delta t", _fmt(epi_diag.violation_ifac))
    _print_kv("Cost constants", "battery=$(epi_diag.battery_cost), pv=$(epi_diag.pv_cost), base_pv_kw=$(epi_diag.base_pv_capacity_kw)")

    println()
    println(rpad("t", 6), rpad("hour", 10), rpad("R chosen", 16), rpad("period cost", 18))
    println(repeat("-", 96))
    for idx in _top_indices(epi_diag.per_period_code; n_show=n_show, largest=true)
        println(
            rpad(string(idx), 6),
            rpad(_fmt(_time_hours(idx, delta_t); digits=3), 10),
            rpad(_fmt(epi_diag.r_argmax[idx]), 16),
            rpad(_fmt(epi_diag.per_period_code[idx]), 18),
        )
    end
end

function verify_saved_result(result_path::AbstractString; solution_idx=nothing, n_show::Int=8)
    result_data = JLD2.load(result_path)
    uncertainty_bounds = result_data["uncertainty_bounds"]
    microgrid_params = result_data["microgrid_params"]
    u_star, gamma_star, selected_idx = _extract_solution(result_data; solution_idx=solution_idx)

    grid_size = Int(microgrid_params["grid_size"])
    n_t = grid_size - 1
    two_batt = haskey(result_data, "two_batt") ? Bool(result_data["two_batt"]) : div(length(u_star), n_t) == 4
    is_cycle = haskey(result_data, "is_cycle") ? Bool(result_data["is_cycle"]) : false

    n_star = _result_float(result_data, "n_star")
    m_star = _result_float(result_data, "m_star"; default=0.0)
    s_star = _result_float(result_data, "s_star")
    pc1, pd1, pc2, pd2 = _unpack_dispatch(u_star, n_t; two_batt=two_batt)

    control_violation = _control_diagnostics(pc1, pd1, pc2, pd2, n_star, m_star, microgrid_params; two_batt=two_batt)
    grid_diag = _grid_diagnostics(pc1, pd1, pc2, pd2, s_star, uncertainty_bounds, microgrid_params)
    soc1_diag = _soc_envelope(pc1, pd1, n_star, uncertainty_bounds, microgrid_params)
    soc2_diag = two_batt ? _soc_envelope(pc2, pd2, m_star, uncertainty_bounds, microgrid_params) : nothing
    epi_diag = _epigraph_diagnostics(grid_diag, result_data, uncertainty_bounds, microgrid_params, n_star, m_star, s_star, gamma_star; two_batt=two_batt)

    _print_section("Saved Result Verification")
    _print_kv("Result file", result_path)
    _print_kv("Selected solution index", selected_idx)
    _print_kv("Formulation", get(result_data, "is_milp", false) ? "MILP" : "NLP")
    _print_kv("two_batt / is_cycle", "$(two_batt) / $(is_cycle)")
    _print_kv("n_star, m_star, s_star", "$(n_star), $(m_star), $(s_star)")
    _print_kv("gamma_star", _fmt(gamma_star))
    _print_kv("scenarios_added", string(get(result_data, "scenarios_added", "unknown")))
    termination_reason = string(get(result_data, "termination_reason", "not recorded"))
    _print_kv("termination_reason", termination_reason)
    _print_kv("last_sigma_star", haskey(result_data, "last_sigma_star") ? _fmt(result_data["last_sigma_star"]) : "not recorded")
    _print_kv("history columns", string(size(result_data["u_star_history"], 2)))
    if termination_reason in ("not recorded", "max_master_scenarios_reached") &&
       get(result_data, "scenarios_added", -1) == size(result_data["u_star_history"], 2)
        _print_kv("capped-run warning", "scenario count equals saved history length; check whether k1_max was reached")
    end

    _print_control_report(control_violation)
    _print_grid_report(grid_diag, microgrid_params; n_show=n_show)
    _print_soc_report("battery 1", soc1_diag, microgrid_params; is_cycle=is_cycle, n_show=n_show)
    if two_batt
        _print_soc_report("battery 2", soc2_diag, microgrid_params; is_cycle=is_cycle, n_show=n_show)
    end
    _print_epigraph_report(epi_diag, gamma_star, microgrid_params; n_show=n_show)

    max_soc_violation = maximum(soc1_diag.violation)
    max_terminal_violation = is_cycle ? max(abs(soc1_diag.soc_min[end] - Float64(microgrid_params["SoC0"])), abs(soc1_diag.soc_max[end] - Float64(microgrid_params["SoC0"]))) : 0.0
    if two_batt
        max_soc_violation = max(max_soc_violation, maximum(soc2_diag.violation))
        if is_cycle
            max_terminal_violation = max(
                max_terminal_violation,
                abs(soc2_diag.soc_min[end] - Float64(microgrid_params["SoC0"])),
                abs(soc2_diag.soc_max[end] - Float64(microgrid_params["SoC0"])),
            )
        end
    end

    _print_section("Bottom Line")
    _print_kv("Control feasible", maximum(control_violation) <= 1e-7)
    _print_kv("Grid robust feasible", maximum(grid_diag.violation) <= 1e-7)
    _print_kv("SoC robust feasible", max_soc_violation <= 1e-7)
    _print_kv("Terminal robust feasible", !is_cycle || max_terminal_violation <= 1e-7)
    _print_kv("Epigraph feasible, legacy cost", epi_diag.violation_code <= 1e-7)
    _print_kv("Epigraph feasible, Delta t cost", epi_diag.violation_ifac <= 1e-7)

    return (
        result_path=result_path,
        solution_idx=selected_idx,
        control_violation=control_violation,
        grid=grid_diag,
        soc1=soc1_diag,
        soc2=soc2_diag,
        epigraph=epi_diag,
        max_soc_violation=max_soc_violation,
        max_terminal_violation=max_terminal_violation,
    )
end

function _parse_cli(args)
    result_path = isempty(args) ? DEFAULT_RESULT_PATH : args[1]
    solution_idx = length(args) >= 2 && lowercase(args[2]) != "last" ? parse(Int, args[2]) : nothing
    n_show = length(args) >= 3 ? parse(Int, args[3]) : 8
    return result_path, solution_idx, n_show
end

if abspath(PROGRAM_FILE) == @__FILE__
    result_path, solution_idx, n_show = _parse_cli(ARGS)
    verify_saved_result(result_path; solution_idx=solution_idx, n_show=n_show)
end
