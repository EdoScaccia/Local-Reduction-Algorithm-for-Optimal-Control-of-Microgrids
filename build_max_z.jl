using JuMP
using Ipopt

function build_max_z(
    is_mip,
    mip_optimizer,
    ipopt_lin_solver, 
    max_iter_ipopt,
    H,
    h,
    grid_size,
    battery_cost,
    pv_cost,
    base_pv_capacity_kw,
    delta_t,
    e_batt,
    is_silent,
    pc_max,
    pd_max,
    pg_max,
    n_max,
    soc0,
    soc_min,
    soc_max,
)
    """
    Build the lowest-level nonlinear maximization problem.
    """
    n_t = grid_size - 1
    cost_scale = max(abs(base_pv_capacity_kw * pv_cost), abs(e_batt * battery_cost), 1.0)
    energy_scale = max(abs(e_batt * n_max), 1.0)
    if is_mip
        if mip_optimizer == "Gurobi"
            model = Model(Gurobi.Optimizer)
        elseif mip_optimizer == "SCIP"
            model = Model(SCIP.Optimizer)
        else
            model = Model(HiGHS.Optimizer)
        end
    else
        model = Model(Ipopt.Optimizer)
        set_attribute(model, "linear_solver", ipopt_lin_solver)
        set_attribute(model, "max_iter", max_iter_ipopt)
        # set_attribute(model, "nlp_scaling_method", "gradient-based")
        # set_attribute(model, "mu_strategy", "adaptive")
    end
    # Set the model to be silent if requested
    if is_silent
        set_silent(model)
    end

    u_dim = 2
    if is_mip
        l_dim_per_t = 2
    else
        l_dim_per_t = 4
    end

    if is_mip
        @variable(model, lambda[1:l_dim_per_t*n_t], Bin)
        @variable(model, s_star_var, Int)
        @variable(model, n_star_var, Int)
    else
        @variable(model, lambda[1:l_dim_per_t*n_t] >= 0)
        @variable(model, n_star_var)
        @variable(model, s_star_var)
    end
    @variable(model, z[1:2*n_t] >= 0)
    @variable(model, u_star_var[1:u_dim*n_t])
    @variable(model, w_star_var[1: 6*n_t])
    @variable(model, sigma_bar)
    @variable(model, sigma_star_var)
    @variable(model, gamma_star_var)
    @variable(model, x[1:grid_size])
    @constraint(model, H * vec(z) .<= h)
    @constraint(model, (x[1] - soc0 * n_star_var * e_batt) / energy_scale == 0)
    @constraint(model, sigma_bar <= sigma_star_var - ((soc_min * n_star_var * e_batt - x[1]) / energy_scale))
    @constraint(model, sigma_bar <= sigma_star_var - ((x[1] - soc_max * n_star_var * e_batt) / energy_scale))

    # Constraints for each time step
    for t in 1:n_t
        u2_idx = u_dim * (t - 1)
        w_idx = 6 * (t - 1)
        z_idx = 2 * (t - 1)
        l_idx = l_dim_per_t * (t - 1)
        # @constraint(
        #     model,
        #     z[z_idx + 1] - z[z_idx + 2] +
        #     u_star_var[u2_idx + 2] - u_star_var[u2_idx + 1] -
        #     w_star_var[w_idx + 1] + s_star_var * w_star_var[w_idx + 2] == 0
        # )
        @constraint(
            model,
            (x[t + 1] - x[t] - (
                w_star_var[w_idx + 5] * u_star_var[u2_idx + 1] -
                w_star_var[w_idx + 6] * u_star_var[u2_idx + 2]
            ) * delta_t) / energy_scale == 0
        )
        @constraint(model, sigma_bar <= sigma_star_var - ((soc_min * n_star_var * e_batt - x[t + 1]) / energy_scale))
        @constraint(model, sigma_bar <= sigma_star_var - ((x[t + 1] - soc_max * n_star_var * e_batt) / energy_scale))

        required_grid_net = (
            w_star_var[w_idx + 1] -
            s_star_var * w_star_var[w_idx + 2] -
            u_star_var[u2_idx + 2] +
            u_star_var[u2_idx + 1]
        )
        @constraint(model, sigma_bar <= sigma_star_var - (required_grid_net - pg_max))
        @constraint(model, sigma_bar <= sigma_star_var - (-required_grid_net - pg_max))

        # Logic constraints
        if is_mip
            @constraint(model, sigma_bar <= sigma_star_var - (-u_star_var[u2_idx + 1]))
            @constraint(model, sigma_bar <= sigma_star_var - (u_star_var[u2_idx + 1] - n_max * pc_max * lambda[l_idx + 1]))
            @constraint(model, sigma_bar <= sigma_star_var - (-u_star_var[u2_idx + 2]))
            @constraint(model, sigma_bar <= sigma_star_var - (u_star_var[u2_idx + 2] - n_max * pd_max * (1 - lambda[l_idx + 1])))
            # Grid
            grid_l_idx = l_idx + 2
            @constraint(model, sigma_bar <= sigma_star_var - (-z[z_idx + 1]))
            @constraint(model, sigma_bar <= sigma_star_var - (z[z_idx + 1] - pg_max * lambda[grid_l_idx]))
            @constraint(model, sigma_bar <= sigma_star_var - (-z[z_idx + 2]))
            @constraint(model, sigma_bar <= sigma_star_var - (z[z_idx + 2] - pg_max * (1 - lambda[grid_l_idx])))
        else
            @constraint(model, lambda[l_idx + 1] + lambda[l_idx + 2] == 1)
            @constraint(model, lambda[l_idx + 3] + lambda[l_idx + 4] == 1)
            @constraint(model, sigma_bar <= sigma_star_var - (lambda[l_idx + 1] * u_star_var[u2_idx + 1] + lambda[l_idx + 2] * u_star_var[u2_idx + 2]))
            @constraint(model, sigma_bar <= sigma_star_var - (lambda[l_idx + 3] * z[z_idx + 1] + lambda[l_idx + 4] * z[z_idx + 2]))
        end
    end

    investment_cost = n_star_var * e_batt * battery_cost + s_star_var * base_pv_capacity_kw * pv_cost
    epigraph_row = (
        investment_cost +
        sum(
            delta_t * (
                w_star_var[6 * (t - 1) + 3] * z[2 * (t - 1) + 1] -
                w_star_var[6 * (t - 1) + 4] * z[2 * (t - 1) + 2]
            )
            for t in 1:n_t
        ) - gamma_star_var
    ) / cost_scale
    @constraint(model, sigma_bar <= sigma_star_var - epigraph_row)

    @objective(model, Max, sigma_bar)

    return model
end
