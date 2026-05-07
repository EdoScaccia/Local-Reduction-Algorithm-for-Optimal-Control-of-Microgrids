using JuMP

include("compute_milp_selector_big_m.jl")

function master_z_selector_row_count(is_mip::Bool, grid_size::Integer)
    n_t = grid_size - 1
    logic_rows_per_t = is_mip ? 8 : 2
    soc_rows = 2 * grid_size
    # soc_rows = 0
    grid_rows = 2 * n_t
    epigraph_rows = 1
    return logic_rows_per_t * n_t + soc_rows + grid_rows + epigraph_rows
end

function _add_master_z_scenario_constraints!(
    master_z,
    z,
    z_scenario,
)
    n_t = div(length(z_scenario), 2)
    for t in 1:n_t
        z_idx = 2 * (t - 1)
        @constraint(master_z, z[z_idx + 1] == z_scenario[z_idx + 1])
        @constraint(master_z, z[z_idx + 2] == z_scenario[z_idx + 2])
    end
    return nothing
end

function add_master_z_scenario!(
    master_z,
    s_blocks,
    scenario_slot,
    z_scenario,
    lambda_scenario,
    scenario_sigma_ubs,
    w,
    sigma,
    n_rows_s,
    u_star,
    gamma_star,
    w_lb,
    w_ub,
    pc_max,
    pd_max,
    pg_max,
    n_max,
    n_star,
    s_star,
    soc0,
    soc_min,
    soc_max,
    battery_cost,
    pv_cost,
    base_pv_capacity_kw,
    delta_t,
    e_batt,
    cost_scale;
    is_mip::Bool=false,
)
    if !isassigned(s_blocks, scenario_slot)
        if is_mip
            s_blocks[scenario_slot] = @variable(master_z, [1:n_rows_s], Bin)
        else
            s_blocks[scenario_slot] = @variable(master_z, [1:n_rows_s], lower_bound = 0.0)
        end
        @constraint(master_z, sum(s_blocks[scenario_slot]) == 1)
    end

    _add_master_z_scenario_constraints!(
        master_z,
        master_z[:z][:, scenario_slot],
        z_scenario,
    )

    g_rows = Any[]
    g_lbs = Float64[]
    g_ubs = Float64[]
    _build_master_z_selector_rows!(
        g_rows,
        g_lbs,
        g_ubs,
        w,
        master_z[:x],
        master_z[:z][:, scenario_slot],
        lambda_scenario,
        u_star,
        gamma_star,
        w_lb,
        w_ub,
        pc_max,
        pd_max,
        pg_max,
        n_max,
        n_star,
        s_star,
        soc0,
        soc_min,
        soc_max,
        battery_cost,
        pv_cost,
        base_pv_capacity_kw,
        delta_t,
        e_batt,
        cost_scale,
        is_mip=is_mip,
    )

    if length(g_rows) != n_rows_s
        error("Mismatch in number of g_i rows: expected $n_rows_s, got $(length(g_rows)).")
    end

    if is_mip
        push!(scenario_sigma_ubs, milp_selector_scenario_sigma_ub(g_ubs))
        sigma_ub = milp_selector_global_sigma_ub(scenario_sigma_ubs)
        _add_milp_selector_constraints!(master_z, sigma, s_blocks[scenario_slot], g_rows, g_lbs, sigma_ub)
    else
        @constraint(
            master_z,
            sum(s_blocks[scenario_slot][i] * (sigma - g_rows[i]) for i in 1:n_rows_s) <= 0,
        )
    end

    return nothing
end

function _add_milp_selector_constraints!(model, sigma, s_block, g_rows, g_lbs, sigma_ub)
    big_m_values = milp_selector_big_m_values(g_lbs, sigma_ub)
    for i in eachindex(g_rows)
        row_big_m = big_m_values[i]
        @constraint(model, sigma <= g_rows[i] + row_big_m * (1 - s_block[i]))
    end
    return nothing
end

function _push_milp_row!(g_rows, g_lbs, g_ubs, expr, lb, ub)
    push!(g_rows, expr)
    push!(g_lbs, Float64(lb))
    push!(g_ubs, Float64(ub))
    return nothing
end

function _build_master_z_selector_rows!(
    g_rows,
    g_lbs,
    g_ubs,
    w,
    x,
    z,
    lambda_scenario,
    u_star,
    gamma_star,
    w_lb,
    w_ub,
    pc_max,
    pd_max,
    pg_max,
    n_max,
    n_star,
    s_star,
    soc0,
    soc_min,
    soc_max,
    battery_cost,
    pv_cost,
    base_pv_capacity_kw,
    delta_t,
    e_batt,
    cost_scale;
    is_mip::Bool=false,
)
    u_dim = 2
    nlp_l_dim_per_t = 4
    milp_l_dim_per_t = 2
    n_t = div(length(u_star), u_dim)
    grid_size = n_t + 1
    energy_scale = max(abs(e_batt * n_star), 1.0)

    if is_mip
        x_lb, x_ub = _soc_state_bounds_from_box(
            w_lb,
            w_ub,
            u_star,
            grid_size,
            soc0,
            n_star,
            delta_t,
            e_batt,
        )
    end

    for t in 1:grid_size
        soc_lb_residual = (soc_min * n_star * e_batt - x[t]) / energy_scale
        soc_ub_residual = (x[t] - soc_max * n_star * e_batt) / energy_scale

        if is_mip
            soc_lb = soc_min * n_star * e_batt
            soc_ub = soc_max * n_star * e_batt
            _push_milp_row!(
                g_rows,
                g_lbs,
                g_ubs,
                soc_lb_residual,
                (soc_lb - x_ub[t]) / energy_scale,
                (soc_lb - x_lb[t]) / energy_scale,
            )
            _push_milp_row!(
                g_rows,
                g_lbs,
                g_ubs,
                soc_ub_residual,
                (x_lb[t] - soc_ub) / energy_scale,
                (x_ub[t] - soc_ub) / energy_scale,
            )
        else
            push!(g_rows, soc_lb_residual)
            push!(g_rows, soc_ub_residual)
        end
    end

    for t in 1:n_t
        w_idx = 6 * (t - 1)
        u_idx = u_dim * (t - 1)
        required_grid_net = (
            w[w_idx + 1, 1] -
            s_star * w[w_idx + 2, 1] -
            u_star[u_idx + 2] +
            u_star[u_idx + 1]
        )
        grid_plus_residual = required_grid_net - pg_max
        grid_minus_residual = -required_grid_net - pg_max

        if is_mip
            d_lb = w_lb[w_idx + 1] - s_star * w_ub[w_idx + 2] - u_star[u_idx + 2] + u_star[u_idx + 1]
            d_ub = w_ub[w_idx + 1] - s_star * w_lb[w_idx + 2] - u_star[u_idx + 2] + u_star[u_idx + 1]
            _push_milp_row!(
                g_rows,
                g_lbs,
                g_ubs,
                grid_plus_residual,
                d_lb - pg_max,
                d_ub - pg_max,
            )
            _push_milp_row!(
                g_rows,
                g_lbs,
                g_ubs,
                grid_minus_residual,
                -d_ub - pg_max,
                -d_lb - pg_max,
            )
        else
            push!(g_rows, grid_plus_residual)
            push!(g_rows, grid_minus_residual)
        end
    end

    for t in 1:n_t
        u_idx = u_dim * (t - 1)
        z_idx = 2 * (t - 1)

        if is_mip
            l_idx = milp_l_dim_per_t * (t - 1)
            _push_milp_row!(g_rows, g_lbs, g_ubs, -u_star[u_idx + 1], -u_star[u_idx + 1], -u_star[u_idx + 1])
            _push_milp_row!(g_rows, g_lbs, g_ubs, u_star[u_idx + 1] - n_max * pc_max * lambda_scenario[l_idx + 1], u_star[u_idx + 1] - n_max * pc_max * lambda_scenario[l_idx + 1], u_star[u_idx + 1] - n_max * pc_max * lambda_scenario[l_idx + 1])
            _push_milp_row!(g_rows, g_lbs, g_ubs, -u_star[u_idx + 2], -u_star[u_idx + 2], -u_star[u_idx + 2])
            _push_milp_row!(g_rows, g_lbs, g_ubs, u_star[u_idx + 2] - n_max * pd_max * (1 - lambda_scenario[l_idx + 1]), u_star[u_idx + 2] - n_max * pd_max * (1 - lambda_scenario[l_idx + 1]), u_star[u_idx + 2] - n_max * pd_max * (1 - lambda_scenario[l_idx + 1]))
            grid_l = l_idx + 2
            _push_milp_row!(g_rows, g_lbs, g_ubs, -z[z_idx + 1], -pg_max, 0.0)
            _push_milp_row!(g_rows, g_lbs, g_ubs, z[z_idx + 1] - pg_max * lambda_scenario[grid_l], -pg_max * lambda_scenario[grid_l], pg_max * (1 - lambda_scenario[grid_l]))
            _push_milp_row!(g_rows, g_lbs, g_ubs, -z[z_idx + 2], -pg_max, 0.0)
            _push_milp_row!(g_rows, g_lbs, g_ubs, z[z_idx + 2] - pg_max * (1 - lambda_scenario[grid_l]), -pg_max * (1 - lambda_scenario[grid_l]), pg_max * lambda_scenario[grid_l])
        else
            l_idx = nlp_l_dim_per_t * (t - 1)
            push!(g_rows, lambda_scenario[l_idx + 1] * u_star[u_idx + 1] + lambda_scenario[l_idx + 2] * u_star[u_idx + 2])
            push!(g_rows, lambda_scenario[l_idx + 3] * z[z_idx + 1] + lambda_scenario[l_idx + 4] * z[z_idx + 2])
        end
    end

    investment_cost = n_star * e_batt * battery_cost + s_star * base_pv_capacity_kw * pv_cost
    epigraph_row = (
        investment_cost +
        sum(
            delta_t * (
                w[6 * (t - 1) + 3, 1] * z[2 * (t - 1) + 1] -
                w[6 * (t - 1) + 4, 1] * z[2 * (t - 1) + 2]
            )
            for t in 1:n_t
        ) - gamma_star
    ) / cost_scale
    if is_mip
        epi_lb = investment_cost - gamma_star
        epi_ub = investment_cost - gamma_star
        for t in 1:n_t
            buy_lb = min(0.0, w_lb[6 * (t - 1) + 3] * pg_max, w_ub[6 * (t - 1) + 3] * pg_max)
            buy_ub = max(0.0, w_lb[6 * (t - 1) + 3] * pg_max, w_ub[6 * (t - 1) + 3] * pg_max)
            sell_lb = min(-w_lb[6 * (t - 1) + 4] * pg_max, -w_ub[6 * (t - 1) + 4] * pg_max, 0.0)
            sell_ub = max(-w_lb[6 * (t - 1) + 4] * pg_max, -w_ub[6 * (t - 1) + 4] * pg_max, 0.0)
            epi_lb += delta_t * (buy_lb + sell_lb)
            epi_ub += delta_t * (buy_ub + sell_ub)
        end
        _push_milp_row!(
            g_rows,
            g_lbs,
            g_ubs,
            epigraph_row,
            epi_lb / cost_scale,
            epi_ub / cost_scale,
        )
    else
        push!(g_rows, epigraph_row)
    end

    return nothing
end

function _soc_state_bounds_from_box(
    w_lb,
    w_ub,
    u_star,
    grid_size,
    soc0,
    n_star,
    delta_t,
    e_batt,
)
    u_dim = 2
    n_t = grid_size - 1
    x_lb = Vector{Float64}(undef, grid_size)
    x_ub = Vector{Float64}(undef, grid_size)
    x0 = soc0 * n_star * e_batt
    x_lb[1] = x0
    x_ub[1] = x0

    for t in 1:n_t
        w_idx = 6 * (t - 1)
        u_idx = u_dim * (t - 1)
        pc = u_star[u_idx + 1]
        pd = u_star[u_idx + 2]
        eta_lb = w_lb[w_idx + 5]
        eta_ub = w_ub[w_idx + 5]
        rho_lb = w_lb[w_idx + 6]
        rho_ub = w_ub[w_idx + 6]
        step_candidates = (
            eta_lb * pc - rho_lb * pd,
            eta_lb * pc - rho_ub * pd,
            eta_ub * pc - rho_lb * pd,
            eta_ub * pc - rho_ub * pd,
        )
        step_lb = delta_t * minimum(step_candidates)
        step_ub = delta_t * maximum(step_candidates)
        x_lb[t + 1] = x_lb[t] + step_lb
        x_ub[t + 1] = x_ub[t] + step_ub
    end

    return x_lb, x_ub
end
