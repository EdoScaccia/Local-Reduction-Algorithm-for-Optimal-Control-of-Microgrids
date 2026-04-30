using JuMP
using HiGHS
using Gurobi
using SCIP

function build_max_z_lp(
    milp_optimizer,
    H,
    h,
    grid_size,
    pc_max,
    pd_max,
    pg_max,
    n_max,
    m_max,
    battery_cost,
    pv_cost,
    base_pv_capacity_kw,
    soc0,
    soc_min,
    soc_max,
    delta_t,
    e_batt,
    energy_scale,
    is_silent;
    two_batt::Bool=true,
    is_cycle::Bool=false,
)
    """
    Build the lowest-level nonlinear maximization problem

        max_{z, lambda, sigma_bar} sigma_bar

    subject to

        sigma_bar <= sigma_star - g_i

    for all master-problem rows g_i built from:
    - logic rows at each time step
    - power-balance equality split into two inequalities at each time step
    - SoC lower/upper rows and optional terminal-cycle rows
    - the epigraph row

    with u_star, w_star, gamma_star, and sigma_star fixed from upper levels.
    """
    n_t = grid_size - 1
    cost_scale = max(abs(base_pv_capacity_kw * pv_cost), abs(e_batt * battery_cost), 1.0)

    if milp_optimizer == "Gurobi"
        model = Model(Gurobi.Optimizer)
    elseif milp_optimizer == "SCIP"
        model = Model(SCIP.Optimizer)
    else
        model = Model(HiGHS.Optimizer)
    end
    if is_silent
        set_silent(model)
    end

    @variable(model, z[1: 2*n_t] >= 0)
    u_dim = two_batt ? 4 : 2
    l_dim_per_t = two_batt ? 3 : 2
    @variable(model, lambda[1:l_dim_per_t*n_t], Bin)
    @variable(model, u_star_var[1:u_dim*n_t])
    @variable(model, w_star_var[1: 6*n_t])
    @variable(model, sigma_bar)
    @variable(model, sigma_star_var)
    @variable(model, gamma_star_var)
    @variable(model, n_star_var, Int)
    if two_batt
        @variable(model, m_star_var, Int)
    end
    @variable(model, s_star_var, Int)

    @constraint(model, H * vec(z) .<= h)

    # Constraints for each time step
    for t in 1:n_t
        u2_idx = u_dim * (t - 1)
        z_idx = 2 * (t - 1)
        l_idx = l_dim_per_t * (t - 1)


        # Logic constraints
        # Battery 1
        @constraint(model, sigma_bar <= sigma_star_var - (-u_star_var[u2_idx + 1]))
        @constraint(model, sigma_bar <= sigma_star_var - (u_star_var[u2_idx + 1] - n_max * pc_max * lambda[l_idx + 1]))
        @constraint(model, sigma_bar <= sigma_star_var - (-u_star_var[u2_idx + 2]))
        @constraint(model, sigma_bar <= sigma_star_var - (u_star_var[u2_idx + 2] - n_max * pd_max * (1 - lambda[l_idx + 1])))
        if two_batt
            # Battery 2
            @constraint(model, sigma_bar <= sigma_star_var - (-u_star_var[u2_idx + 3]))
            @constraint(model, sigma_bar <= sigma_star_var - (u_star_var[u2_idx + 3] - m_max * pc_max * lambda[l_idx + 2]))
            @constraint(model, sigma_bar <= sigma_star_var - (-u_star_var[u2_idx + 4]))
            @constraint(model, sigma_bar <= sigma_star_var - (u_star_var[u2_idx + 4] - m_max * pd_max * (1 - lambda[l_idx + 2])))
        end
        # Grid
        grid_l_idx = l_idx + (two_batt ? 3 : 2)
        @constraint(model, sigma_bar <= sigma_star_var - (-z[z_idx + 1]))
        @constraint(model, sigma_bar <= sigma_star_var - (z[z_idx + 1] - pg_max * lambda[grid_l_idx]))
        @constraint(model, sigma_bar <= sigma_star_var - (-z[z_idx + 2]))
        @constraint(model, sigma_bar <= sigma_star_var - (z[z_idx + 2] - pg_max * (1 - lambda[grid_l_idx])))

        idx = 6 * (t - 1)
        PL_t = w_star_var[idx + 1]
        PR_t = w_star_var[idx + 2]
        batt2_net_t = two_batt ? u_star_var[u2_idx + 4] - u_star_var[u2_idx + 3] : 0.0
        @constraint(
            model, sigma_bar <= sigma_star_var -
            (z[z_idx + 1] - z[z_idx + 2] + u_star_var[u2_idx + 2] - u_star_var[u2_idx + 1] + batt2_net_t - PL_t + s_star_var * PR_t))
        @constraint(
            model, sigma_bar <= sigma_star_var +
            (z[z_idx + 1] - z[z_idx + 2] + u_star_var[u2_idx + 2] - u_star_var[u2_idx + 1] + batt2_net_t - PL_t + s_star_var * PR_t))
    end

    x1 = Vector{Any}(undef, grid_size)
    x2 = two_batt ? Vector{Any}(undef, grid_size) : Any[]
    x1[1] = soc0 * n_star_var * e_batt
    if two_batt
        x2[1] = soc0 * m_star_var * e_batt
    end
    for t in 1:n_t
        u_idx = u_dim * (t - 1)
        w_idx = 6 * (t - 1)
        eta_ch_t = w_star_var[w_idx + 5]
        rho_dis_t = w_star_var[w_idx + 6]
        x1[t + 1] = x1[t] + (eta_ch_t * u_star_var[u_idx + 1] - rho_dis_t * u_star_var[u_idx + 2]) * delta_t
        if two_batt
            x2[t + 1] = x2[t] + (eta_ch_t * u_star_var[u_idx + 3] - rho_dis_t * u_star_var[u_idx + 4]) * delta_t
        end
    end

    for t in 1:grid_size
        @constraint(model, sigma_bar <= sigma_star_var - ((soc_min * n_star_var * e_batt - x1[t]) / energy_scale))
        @constraint(model, sigma_bar <= sigma_star_var - ((x1[t] - soc_max * n_star_var * e_batt) / energy_scale))
        if two_batt
            @constraint(model, sigma_bar <= sigma_star_var - ((soc_min * m_star_var * e_batt - x2[t]) / energy_scale))
            @constraint(model, sigma_bar <= sigma_star_var - ((x2[t] - soc_max * m_star_var * e_batt) / energy_scale))
        end
    end
    if is_cycle
        terminal1 = (x1[grid_size] - soc0 * n_star_var * e_batt) / energy_scale
        @constraint(model, sigma_bar <= sigma_star_var - terminal1)
        @constraint(model, sigma_bar <= sigma_star_var + terminal1)
        if two_batt
            terminal2 = (x2[grid_size] - soc0 * m_star_var * e_batt) / energy_scale
            @constraint(model, sigma_bar <= sigma_star_var - terminal2)
            @constraint(model, sigma_bar <= sigma_star_var + terminal2)
        end
    end

    battery_count_expr = n_star_var
    if two_batt
        battery_count_expr += m_star_var
    end
    epi_expr = (
        delta_t * sum(
            w_star_var[6 * (t - 1) + 3] * z[2 * (t - 1) + 1] -
            w_star_var[6 * (t - 1) + 4] * z[2 * (t - 1) + 2]
            for t in 1:n_t
        ) +
        battery_count_expr * e_batt * battery_cost +
        s_star_var * base_pv_capacity_kw * pv_cost -
        gamma_star_var
    ) / cost_scale
    @constraint(model, sigma_bar <= sigma_star_var - epi_expr)

    @objective(model, Max, sigma_bar)

    return model
end

function build_max_z_lp_fixed(
    milp_optimizer,
    H,
    h,
    grid_size,
    pc_max,
    pd_max,
    pg_max,
    n_max,
    m_max,
    battery_cost,
    pv_cost,
    base_pv_capacity_kw,
    soc0,
    soc_min,
    soc_max,
    delta_t,
    e_batt,
    energy_scale,
    u_star,
    w_star,
    gamma_star,
    sigma_star,
    n_star,
    m_star,
    s_star,
    is_silent;
    two_batt::Bool=true,
    is_cycle::Bool=false,
)
    n_t = grid_size - 1
    cost_scale = max(abs(base_pv_capacity_kw * pv_cost), abs(e_batt * battery_cost), 1.0)

    if milp_optimizer == "Gurobi"
        model = Model(Gurobi.Optimizer)
    elseif milp_optimizer == "SCIP"
        model = Model(SCIP.Optimizer)
    else
        model = Model(HiGHS.Optimizer)
    end
    is_silent && set_silent(model)

    @variable(model, z[1:2*n_t] >= 0)
    u_dim = two_batt ? 4 : 2
    l_dim_per_t = two_batt ? 3 : 2
    @variable(model, lambda[1:l_dim_per_t*n_t], Bin)
    @variable(model, sigma_bar)
    @constraint(model, H * vec(z) .<= h)

    for t in 1:n_t
        u_idx = u_dim * (t - 1)
        z_idx = 2 * (t - 1)
        l_idx = l_dim_per_t * (t - 1)
        w_idx = 6 * (t - 1)

        @constraint(model, sigma_bar <= sigma_star - (-u_star[u_idx + 1]))
        @constraint(model, sigma_bar <= sigma_star - (u_star[u_idx + 1] - n_max * pc_max * lambda[l_idx + 1]))
        @constraint(model, sigma_bar <= sigma_star - (-u_star[u_idx + 2]))
        @constraint(model, sigma_bar <= sigma_star - (u_star[u_idx + 2] - n_max * pd_max * (1 - lambda[l_idx + 1])))

        if two_batt
            @constraint(model, sigma_bar <= sigma_star - (-u_star[u_idx + 3]))
            @constraint(model, sigma_bar <= sigma_star - (u_star[u_idx + 3] - m_max * pc_max * lambda[l_idx + 2]))
            @constraint(model, sigma_bar <= sigma_star - (-u_star[u_idx + 4]))
            @constraint(model, sigma_bar <= sigma_star - (u_star[u_idx + 4] - m_max * pd_max * (1 - lambda[l_idx + 2])))
        end

        grid_l_idx = l_idx + (two_batt ? 3 : 2)
        @constraint(model, sigma_bar <= sigma_star - (-z[z_idx + 1]))
        @constraint(model, sigma_bar <= sigma_star - (z[z_idx + 1] - pg_max * lambda[grid_l_idx]))
        @constraint(model, sigma_bar <= sigma_star - (-z[z_idx + 2]))
        @constraint(model, sigma_bar <= sigma_star - (z[z_idx + 2] - pg_max * (1 - lambda[grid_l_idx])))

        batt2_net_t = two_batt ? u_star[u_idx + 4] - u_star[u_idx + 3] : 0.0
        bal = z[z_idx + 1] - z[z_idx + 2] + u_star[u_idx + 2] - u_star[u_idx + 1] +
              batt2_net_t - w_star[w_idx + 1] + s_star * w_star[w_idx + 2]
        @constraint(model, sigma_bar <= sigma_star - bal)
        @constraint(model, sigma_bar <= sigma_star + bal)
    end

    x1 = Vector{Any}(undef, grid_size)
    x2 = two_batt ? Vector{Any}(undef, grid_size) : Any[]
    x1[1] = soc0 * n_star * e_batt
    if two_batt
        x2[1] = soc0 * m_star * e_batt
    end
    for t in 1:n_t
        u_idx = u_dim * (t - 1)
        w_idx = 6 * (t - 1)
        eta_ch_t = w_star[w_idx + 5]
        rho_dis_t = w_star[w_idx + 6]
        x1[t + 1] = x1[t] + (eta_ch_t * u_star[u_idx + 1] - rho_dis_t * u_star[u_idx + 2]) * delta_t
        if two_batt
            x2[t + 1] = x2[t] + (eta_ch_t * u_star[u_idx + 3] - rho_dis_t * u_star[u_idx + 4]) * delta_t
        end
    end

    for t in 1:grid_size
        @constraint(model, sigma_bar <= sigma_star - ((soc_min * n_star * e_batt - x1[t]) / energy_scale))
        @constraint(model, sigma_bar <= sigma_star - ((x1[t] - soc_max * n_star * e_batt) / energy_scale))
        if two_batt
            @constraint(model, sigma_bar <= sigma_star - ((soc_min * m_star * e_batt - x2[t]) / energy_scale))
            @constraint(model, sigma_bar <= sigma_star - ((x2[t] - soc_max * m_star * e_batt) / energy_scale))
        end
    end
    if is_cycle
        terminal1 = (x1[grid_size] - soc0 * n_star * e_batt) / energy_scale
        @constraint(model, sigma_bar <= sigma_star - terminal1)
        @constraint(model, sigma_bar <= sigma_star + terminal1)
        if two_batt
            terminal2 = (x2[grid_size] - soc0 * m_star * e_batt) / energy_scale
            @constraint(model, sigma_bar <= sigma_star - terminal2)
            @constraint(model, sigma_bar <= sigma_star + terminal2)
        end
    end

    epi_expr = (
        delta_t * sum(
            w_star[6 * (t - 1) + 3] * z[2 * (t - 1) + 1] -
            w_star[6 * (t - 1) + 4] * z[2 * (t - 1) + 2]
            for t in 1:n_t
        ) +
        (n_star + (two_batt ? m_star : 0.0)) * e_batt * battery_cost +
        s_star * base_pv_capacity_kw * pv_cost -
        gamma_star
    ) / cost_scale
    @constraint(model, sigma_bar <= sigma_star - epi_expr)
    @objective(model, Max, sigma_bar)

    return model
end
