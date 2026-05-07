using JuMP
using LinearAlgebra
using Ipopt
using HiGHS
using Gurobi
using SCIP

function build_z_master(
    is_mip,
    mip_optimizer,
    ipopt_lin_solver,
    max_iter_ipopt,
    s_blocks,
    grid_size,
    P,
    p,
    H,
    h,
    k2_max,
    soc0,
    soc_min,
    soc_max,
    n_star,
    u_star,
    s_star,
    delta_t,
    e_batt,
    is_silent;
    is_cycle::Bool=false,
)
    """
    Build the finite-scenario z-master using the exact lambda-smoothed
    disjunction reformulation.

    For sampled z and lambda-scenarios z^k and lambda^k, this solves

    max_{w in W, sigma, s_k} sigma
    s.t. forall k in {1,...,k2_max}:
        sum_i s_{j,k} * (sigma - g_i) <= 0
    
    SOC bounds, dynamics, and initial/terminal equalities are enforced directly.

    This problem is solved with fixed scenarios z and lambda with associated existence variables s.
    This is a problem with bilinear constraints and so it is locally solved using Ipopt.
    """
    # Get variables dimensions
    n_rows_w = size(P, 2)
    n_t = grid_size - 1
    u_dim = 2
    n_rows_s = master_z_selector_row_count(is_mip, grid_size)
    w_ub = p[1:n_rows_w]
    w_lb = -p[n_rows_w + 1:2 * n_rows_w]
    w_mid = 0.5 .* (w_lb .+ w_ub)
    energy_scale = max(abs(e_batt * n_star), 1.0)

    # Create model
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

    # Create variables.
    @variable(model, w[1:n_rows_w, 1:1])
    @variable(model, z[1:2*n_t, 1:k2_max] >= 0)
    x_dim = grid_size
    @variable(model, x[1:x_dim])
    if is_mip
        s_blocks[1] = @variable(model, [1:n_rows_s], Bin)
    else
        s_blocks[1] = @variable(model, [1:n_rows_s], lower_bound=0)
    end
    @variable(model, sigma)
    @constraint(model, sum(s_blocks[1]) == 1)

    for i in 1:n_rows_w
        set_start_value(w[i, 1], w_mid[i])
    end
    for t in 1:grid_size
        set_start_value(x[t], soc0 * n_star * e_batt)
    end
    for t in 1:n_t
        for k in 1:k2_max
            set_start_value(z[2 * (t - 1) + 1, k], 0.0)
            set_start_value(z[2 * (t - 1) + 2, k], 0.0)
        end
    end
    set_start_value(sigma, 0.0)

    # Constraint on uncertainty
    @constraint(model, P * w - p .<= 0)
    for k in 1:k2_max
        @constraint(model, H * z[:, k] .<= h)
    end
    # @constraint(model, [t in 1:grid_size], (x[t] - soc_min * n_star * e_batt) / energy_scale >= 0)
    # @constraint(model, [t in 1:grid_size], (x[t] - soc_max * n_star * e_batt) / energy_scale <= 0)
    @constraint(model, (x[1] - soc0 * n_star * e_batt) / energy_scale == 0)
    if is_cycle
        @constraint(model, (x[grid_size] - soc0 * n_star * e_batt) / energy_scale == 0)
    end

    for t in 1:n_t
        w_idx = 6 * (t - 1)
        u_idx = u_dim * (t - 1)
        eta_ch_t = w[w_idx + 5, 1]
        rho_dis_t = w[w_idx + 6, 1]
        @constraint(
            model,
            (x[t + 1] - x[t] - (eta_ch_t * u_star[u_idx + 1] - rho_dis_t * u_star[u_idx + 2]) * delta_t) / energy_scale == 0
        )
    end

    # Objective is to maximise sigma (the violation of the current master solution) 
    @objective(model, Max, sigma)

    return model, s_blocks
end
