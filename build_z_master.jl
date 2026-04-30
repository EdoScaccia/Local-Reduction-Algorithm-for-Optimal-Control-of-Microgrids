using JuMP
using LinearAlgebra
using Ipopt
using HSL_jll

if !(@isdefined local_reduction_g_row_count)
    include("local_reduction_state_rows.jl")
end

function build_z_master(
    s_blocks,
    ipopt_lin_solver,
    max_iter_ipopt,
    grid_size,
    P,
    p,
    soc0,
    soc_min,
    soc_max,
    n_star,
    m_star,
    u_star,
    delta_t,
    e_batt,
    is_silent;
    two_batt::Bool=true,
    is_cycle::Bool=false,
)
    """
    Build the finite-scenario z-master using the exact lambda-smoothed
    disjunction reformulation.

    For sampled z and lambda-scenarios z^k and lambda^k, this solves

    max_{w in W, sigma, s_k} sigma
    s.t. forall k in {1,...,k2_max}:
        sum_i s_{j,k} * (sigma - g_i) <= 0
    
    This problem is solved with fixed scenarios z and lambda with associated existence variables s.
    This is a problem with bilinear constraints and so it is locally solved using Ipopt.
    """
    # Get variables dimensions
    n_rows_w = size(P, 2)
    n_t = grid_size - 1
    u_dim = two_batt ? 4 : 2
    n_rows_s = local_reduction_g_row_count(grid_size; two_batt=two_batt, is_milp=false, is_cycle=is_cycle)
    w_ub = p[1:n_rows_w]
    w_lb = -p[n_rows_w + 1:2 * n_rows_w]
    w_mid = 0.5 .* (w_lb .+ w_ub)
    energy_scale = max(abs(e_batt * max(n_star, two_batt ? m_star : n_star)), 1.0)

    # Create Ipopt model
    model = Model(Ipopt.Optimizer)
    set_attribute(model, "linear_solver", ipopt_lin_solver)
    set_attribute(model, "max_iter", max_iter_ipopt)
    set_attribute(model, "nlp_scaling_method", "gradient-based")
    set_attribute(model, "mu_strategy", "adaptive")
    set_attribute(model, "bound_push", 1e-8)
    set_attribute(model, "bound_frac", 1e-8)
    if is_silent
        set_silent(model)
    end

    # Create variables.
    @variable(model, w[1:n_rows_w, 1:1])
    x_dim = two_batt ? 2 * grid_size : grid_size
    @variable(model, x[1:x_dim])
    s_blocks[1] = @variable(model, [1:n_rows_s], lower_bound = 0.0)
    @variable(model, sigma)

    for i in 1:n_rows_w
        set_start_value(w[i, 1], w_mid[i])
    end
    for t in 1:grid_size
        set_start_value(x[t], soc0 * n_star * e_batt)
        if two_batt
            set_start_value(x[grid_size + t], soc0 * m_star * e_batt)
        end
    end
    set_start_value(sigma, 0.0)

    # Constraint on uncertainty
    @constraint(model, P * w - p .<= 0)
    @constraint(model, (x[1] - soc0 * n_star * e_batt) / energy_scale == 0)
    if two_batt
        @constraint(model, (x[grid_size + 1] - soc0 * m_star * e_batt) / energy_scale == 0)
    end
    # Simplex constraint on s
    @constraint(model, sum(s_blocks[1]) == 1)

    for t in 1:n_t
        w_idx = 6 * (t - 1)
        u_idx = u_dim * (t - 1)
        eta_ch_t = w[w_idx + 5, 1]
        rho_dis_t = w[w_idx + 6, 1]
        @constraint(
            model,
            (x[t + 1] - x[t] - (eta_ch_t * u_star[u_idx + 1] - rho_dis_t * u_star[u_idx + 2]) * delta_t) / energy_scale == 0
        )
        if two_batt
            @constraint(
                model,
                (x[grid_size + t + 1] - x[grid_size + t] - (eta_ch_t * u_star[u_idx + 3] - rho_dis_t * u_star[u_idx + 4]) * delta_t) / energy_scale == 0
            )
        end
    end

    # Objective is to maximise sigma (the violation of the current master solution) 
    @objective(model, Max, sigma)

    return model, s_blocks
end
