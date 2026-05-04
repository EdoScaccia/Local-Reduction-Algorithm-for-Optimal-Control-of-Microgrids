using JuMP
using Ipopt
using HiGHS
using Gurobi
using SCIP

function _make_epi_optimizer(
    optimizer_name;
    ipopt_lin_solver="ma97",
    max_iter_ipopt=1000,
    is_silent=true,
)
    if optimizer_name == "Gurobi"
        model = Model(Gurobi.Optimizer)
        set_optimizer_attribute(model, "NonConvex", 2)
    elseif optimizer_name == "SCIP"
        model = Model(SCIP.Optimizer)
    elseif optimizer_name == "HiGHS"
        model = Model(HiGHS.Optimizer)
    else
        model = Model(Ipopt.Optimizer)
        set_attribute(model, "linear_solver", ipopt_lin_solver)
        set_attribute(model, "max_iter", max_iter_ipopt)
        set_attribute(model, "nlp_scaling_method", "gradient-based")
        set_attribute(model, "mu_strategy", "adaptive")
    end
    is_silent && set_silent(model)
    return model
end

function build_max_epi(
    optimizer_name,
    grid_size,
    P,
    p,
    H,
    h,
    soc0,
    soc_min,
    soc_max,
    pc_max,
    pd_max,
    pg_max,
    n_max,
    m_max,
    u_star,
    gamma_star,
    n_star,
    m_star,
    s_star,
    battery_cost,
    pv_cost,
    base_pv_capacity_kw,
    delta_t,
    e_batt,
    is_silent;
    two_batt::Bool=true,
    is_cycle::Bool=false,
    ipopt_lin_solver="ma97",
    max_iter_ipopt=1000,
)
    """
    Build the separate robust epigraph separation problem.

    Given a fixed master solution (u_star, n_star, m_star, s_star, gamma_star),
    this solves

        max_{w,z,x} J(w,z) - gamma_star

    subject to uncertainty bounds, grid bounds, load balance, and battery
    dynamics/SoC constraints. The grid import/export and battery
    charge/discharge logic are enforced inside this oracle as feasibility
    constraints, not as local-reduction rows.

    The objective is scaled by the same cost scale used elsewhere.
    """
    n_rows_w = size(P, 2)
    n_t = grid_size - 1
    u_dim = two_batt ? 4 : 2
    x_dim = two_batt ? 2 * grid_size : grid_size
    w_ub = p[1:n_rows_w]
    w_lb = -p[n_rows_w + 1:2 * n_rows_w]
    w_mid = 0.5 .* (w_lb .+ w_ub)
    energy_scale = max(abs(e_batt * max(n_star, two_batt ? m_star : n_star)), 1.0)
    cost_scale = max(abs(base_pv_capacity_kw * pv_cost), abs(e_batt * battery_cost), 1.0)

    model = _make_epi_optimizer(
        optimizer_name;
        ipopt_lin_solver=ipopt_lin_solver,
        max_iter_ipopt=max_iter_ipopt,
        is_silent=is_silent,
    )

    # Worst-case uncertainty, grid import/export, and reconstructed battery
    # energy trajectories.
    @variable(model, w[1:n_rows_w, 1:1])
    @variable(model, z[1:2*n_t] >= 0)
    @variable(model, x[1:x_dim])
    if optimizer_name in ("Gurobi", "SCIP")
        @variable(model, lambda_grid[1:n_t], Bin)
        @variable(model, lambda_batt1[1:n_t], Bin)
        if two_batt
            @variable(model, lambda_batt2[1:n_t], Bin)
        end
    else
        @variable(model, 0 <= lambda_grid[1:n_t] <= 1)
        @variable(model, 0 <= lambda_batt1[1:n_t] <= 1)
        if two_batt
            @variable(model, 0 <= lambda_batt2[1:n_t] <= 1)
        end
    end
    @variable(model, u_star_var[1:u_dim*n_t])
    @variable(model, gamma_star_var)
    @variable(model, n_star_var)
    if two_batt
        @variable(model, m_star_var)
    end
    @variable(model, s_star_var)

    fix.(u_star_var, u_star, force=true)
    fix(gamma_star_var, gamma_star, force=true)
    fix(n_star_var, n_star, force=true)
    if two_batt
        fix(m_star_var, m_star, force=true)
    end
    fix(s_star_var, s_star, force=true)

    # Deterministic starts are only warm starts; feasibility is enforced below.
    for i in 1:n_rows_w
        set_start_value(w[i, 1], w_mid[i])
    end
    for t in 1:n_t
        set_start_value(z[2 * (t - 1) + 1], 0.0)
        set_start_value(z[2 * (t - 1) + 2], 0.0)
    end
    for t in 1:grid_size
        set_start_value(x[t], soc0 * n_star * e_batt)
        if two_batt
            set_start_value(x[grid_size + t], soc0 * m_star * e_batt)
        end
    end

    # Uncertainty and grid-cap constraints.
    @constraint(model, P * w - p .<= 0)
    @constraint(model, H * vec(z) .<= h)
    @constraint(model, [t in 1:n_t], z[2 * (t - 1) + 1] <= pg_max * lambda_grid[t])
    @constraint(model, [t in 1:n_t], z[2 * (t - 1) + 2] <= pg_max * (1 - lambda_grid[t]))

    # Fixed control policy must be compatible with a charge/discharge mode.
    @constraint(model, [t in 1:n_t], u_star_var[u_dim * (t - 1) + 1] <= n_max * pc_max * lambda_batt1[t])
    @constraint(model, [t in 1:n_t], u_star_var[u_dim * (t - 1) + 2] <= n_max * pd_max * (1 - lambda_batt1[t]))
    if two_batt
        @constraint(model, [t in 1:n_t], u_star_var[u_dim * (t - 1) + 3] <= m_max * pc_max * lambda_batt2[t])
        @constraint(model, [t in 1:n_t], u_star_var[u_dim * (t - 1) + 4] <= m_max * pd_max * (1 - lambda_batt2[t]))
    end

    # Battery 1 energy limits and initial/optional cyclic terminal condition.
    @constraint(model, [t in 1:grid_size], (x[t] - soc_min * n_star_var * e_batt) / energy_scale >= 0)
    @constraint(model, [t in 1:grid_size], (x[t] - soc_max * n_star_var * e_batt) / energy_scale <= 0)
    @constraint(model, (x[1] - soc0 * n_star_var * e_batt) / energy_scale == 0)
    if is_cycle
        @constraint(model, (x[grid_size] - soc0 * n_star_var * e_batt) / energy_scale == 0)
    end

    if two_batt
        # Battery 2 energy limits and initial/optional cyclic terminal condition.
        @constraint(model, [t in 1:grid_size], (x[grid_size + t] - soc_min * m_star_var * e_batt) / energy_scale >= 0)
        @constraint(model, [t in 1:grid_size], (x[grid_size + t] - soc_max * m_star_var * e_batt) / energy_scale <= 0)
        @constraint(model, (x[grid_size + 1] - soc0 * m_star_var * e_batt) / energy_scale == 0)
        if is_cycle
            @constraint(model, (x[2 * grid_size] - soc0 * m_star_var * e_batt) / energy_scale == 0)
        end
    end

    for t in 1:n_t
        w_idx = 6 * (t - 1)
        u_idx = u_dim * (t - 1)
        z_idx = 2 * (t - 1)
        eta_ch_t = w[w_idx + 5, 1]
        rho_dis_t = w[w_idx + 6, 1]
        batt2_net_t = two_batt ? u_star_var[u_idx + 4] - u_star_var[u_idx + 3] : 0.0

        # Grid import/export is a recourse split satisfying load balance.
        @constraint(
            model,
            z[z_idx + 1] - z[z_idx + 2] + u_star_var[u_idx + 2] - u_star_var[u_idx + 1] +
            batt2_net_t - w[w_idx + 1, 1] + s_star_var * w[w_idx + 2, 1] == 0
        )

        # Battery dynamics are uniquely reconstructed from fixed controls and
        # uncertain efficiencies.
        @constraint(
            model,
            (x[t + 1] - x[t] - (eta_ch_t * u_star_var[u_idx + 1] - rho_dis_t * u_star_var[u_idx + 2]) * delta_t) / energy_scale == 0
        )
        if two_batt
            @constraint(
                model,
                (x[grid_size + t + 1] - x[grid_size + t] - (eta_ch_t * u_star_var[u_idx + 3] - rho_dis_t * u_star_var[u_idx + 4]) * delta_t) / energy_scale == 0
            )
        end
    end

    # Robust epigraph violation. A positive optimum means gamma_star is too low
    # for at least one feasible uncertainty/recourse realization.
    epi_violation = (
        sum(
            delta_t * (
                w[6 * (t - 1) + 3, 1] * z[2 * (t - 1) + 1] -
                w[6 * (t - 1) + 4, 1] * z[2 * (t - 1) + 2]
            )
            for t in 1:n_t
        ) +
        (n_star_var + (two_batt ? m_star_var : 0.0)) * e_batt * battery_cost +
        s_star_var * base_pv_capacity_kw * pv_cost -
        gamma_star_var
    ) / cost_scale

    @objective(model, Max, epi_violation)

    return model
end
