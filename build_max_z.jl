using JuMP
using Ipopt

function build_max_z(
    ipopt_lin_solver, 
    max_iter_ipopt,
    H,
    h,
    grid_size,
    battery_cost,
    pv_cost,
    base_pv_capacity_kw,
    e_batt,
    is_silent;
    two_batt::Bool=true,
)
    """
    Build the lowest-level nonlinear maximization problem

        max_{z, lambda, sigma_bar} sigma_bar

    subject to

        sigma_bar <= sigma_star - g_i

    for all master-problem rows g_i built from:
    - logic rows at each time step
    - power-balance equality split into two inequalities at each time step
    - the epigraph row

    with u_star, w_star, gamma_star, and sigma_star fixed from upper levels.
    """
    n_t = grid_size - 1
    cost_scale = max(abs(base_pv_capacity_kw * pv_cost), abs(e_batt * battery_cost), 1.0)

    model = Model(Ipopt.Optimizer)
    set_attribute(model, "linear_solver", ipopt_lin_solver)
    set_attribute(model, "max_iter", max_iter_ipopt)
    set_attribute(model, "nlp_scaling_method", "gradient-based")
    set_attribute(model, "mu_strategy", "adaptive")
    if is_silent
        set_silent(model)
    end

    @variable(model, z[1: 2*n_t] >= 0)
    u_dim = two_batt ? 4 : 2
    l_dim_per_t = two_batt ? 6 : 4
    @variable(model, lambda[1:l_dim_per_t*n_t] >= 0)
    @variable(model, u_star_var[1:u_dim*n_t])
    @variable(model, w_star_var[1: 6*n_t])
    @variable(model, sigma_bar)
    @variable(model, sigma_star_var)
    @variable(model, gamma_star_var)
    @variable(model, n_star_var)
    if two_batt
        @variable(model, m_star_var)
    end
    @variable(model, s_star_var)

    @constraint(model, H * vec(z) .<= h)

    # Constraints for each time step
    for t in 1:n_t
        u2_idx = u_dim * (t - 1)
        z_idx = 2 * (t - 1)
        l_idx = l_dim_per_t * (t - 1)

        # Simplex constraints on lambda
        @constraint(model, lambda[l_idx + 1] + lambda[l_idx + 2] == 1)
        @constraint(model, lambda[l_idx + 3] + lambda[l_idx + 4] == 1)
        if two_batt
            @constraint(model, lambda[l_idx + 5] + lambda[l_idx + 6] == 1)
        end

        # Logic constraints
        @constraint(model, sigma_bar <= sigma_star_var - (lambda[l_idx + 1] * u_star_var[u2_idx + 1] + lambda[l_idx + 2] * u_star_var[u2_idx + 2]))
        @constraint(model, sigma_bar <= sigma_star_var - (lambda[l_idx + 3] * z[z_idx + 1] + lambda[l_idx + 4] * z[z_idx + 2]))
        if two_batt
            @constraint(model, sigma_bar <= sigma_star_var - (lambda[l_idx + 5] * u_star_var[u2_idx + 3] + lambda[l_idx + 6] * u_star_var[u2_idx + 4]))
        end

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

    battery_count_expr = n_star_var
    if two_batt
        battery_count_expr += m_star_var
    end
    epi_expr = (
        sum(
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
