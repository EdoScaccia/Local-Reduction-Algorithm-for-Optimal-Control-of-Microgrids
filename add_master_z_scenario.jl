using JuMP

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
