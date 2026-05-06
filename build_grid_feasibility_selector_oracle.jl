using JuMP
using Ipopt
using HiGHS
using Gurobi
using SCIP

function build_grid_feasibility_selector_oracle(
    is_mip,
    mip_optimizer,
    ipopt_lin_solver, 
    max_iter_ipopt,
    P,
    p,
    grid_size,
    pg_max;
    pc_max=nothing,
    pd_max=nothing,
    n_max=nothing,
    s_max=nothing,
    is_silent::Bool=true,
)
    n_w = size(P, 2)
    n_t = grid_size - 1
    u_dim = 2
    w_ub = p[1:n_w]
    w_lb = -p[n_w + 1:2*n_w]
    u_pc_ub = (pc_max === nothing || n_max === nothing) ? Inf : n_max * pc_max
    u_pd_ub = (pd_max === nothing || n_max === nothing) ? Inf : n_max * pd_max
    s_ub = s_max === nothing ? Inf : s_max
    if !all(isfinite, (u_pc_ub, u_pd_ub, s_ub))
        error("Finite pc_max, pd_max, n_max, and s_max are required to build the reusable grid selector oracle.")
    end

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

    @variable(model, w[1:n_w])
    @variable(model, u_star_var[1:u_dim*n_t])
    @variable(model, s_star_var)
    @variable(model, sigma)
    if is_mip   
        @variable(model, y[1:2*n_t], Bin)
    else
        @variable(model, y[1:2*n_t], lower_bound=0)
    end
    @constraint(model, P * w .<= p)
    @constraint(model, sum(y) == 1)
    for t in 1:n_t
        u_idx = u_dim * (t - 1)
        set_lower_bound(u_star_var[u_idx + 1], 0.0)
        set_lower_bound(u_star_var[u_idx + 2], 0.0)
        isfinite(u_pc_ub) && set_upper_bound(u_star_var[u_idx + 1], u_pc_ub)
        isfinite(u_pd_ub) && set_upper_bound(u_star_var[u_idx + 2], u_pd_ub)
    end
    set_lower_bound(s_star_var, 1.0)
    isfinite(s_ub) && set_upper_bound(s_star_var, s_ub)

    g_rows = Any[]
    g_lbs = Float64[]
    g_ubs = Float64[]

    for t in 1:n_t
        w_idx = 6 * (t - 1)
        u_idx = u_dim * (t - 1)

        d_t = w[w_idx + 1] - s_star_var * w[w_idx + 2] -
              u_star_var[u_idx + 2] + u_star_var[u_idx + 1]
        push!(g_rows, d_t - pg_max)
        push!(g_rows, -d_t - pg_max)

        s_pr_candidates = (
            1.0 * w_lb[w_idx + 2],
            1.0 * w_ub[w_idx + 2],
            s_ub * w_lb[w_idx + 2],
            s_ub * w_ub[w_idx + 2],
        )
        s_pr_lb = minimum(s_pr_candidates)
        s_pr_ub = maximum(s_pr_candidates)
        d_lb = w_lb[w_idx + 1] - s_pr_ub - u_pd_ub
        d_ub = w_ub[w_idx + 1] - s_pr_lb + u_pc_ub

        push!(g_lbs, d_lb - pg_max)
        push!(g_ubs, d_ub - pg_max)
        push!(g_lbs, -d_ub - pg_max)
        push!(g_ubs, -d_lb - pg_max)
    end

    if is_mip
        sigma_ub = maximum(g_ubs)
        big_m_values = max.(0.0, sigma_ub .- g_lbs)
        for i in eachindex(g_rows)
            @constraint(model, sigma <= g_rows[i] + big_m_values[i] * (1 - y[i]))
        end
    else
        @constraint(model, sum(y[i] * (sigma - g_rows[i]) for i in eachindex(g_rows)) <= 0)
    end
    @objective(model, Max, sigma)

    return model
end
