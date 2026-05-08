include("build_z_master.jl")
include("assert_solution.jl")
include("solve_master_z_with_restarts.jl")
include("initialize_master_z.jl")
include("compute_milp_selector_big_m.jl")

function _z_master_g_row_values(
    z_scenario,
    lambda_scenario,
    w_scenario,
    u_star,
    gamma_star,
    n_star,
    s_star,
    soc0,
    soc_min,
    soc_max,
    pg_max,
    battery_cost,
    pv_cost,
    base_pv_capacity_kw,
    e_batt;
    delta_t=1.0,
)
    u_dim = 2
    l_dim_per_t = 4
    n_t = div(length(u_star), u_dim)
    grid_size = n_t + 1
    cost_scale = max(abs(base_pv_capacity_kw * pv_cost), abs(e_batt * battery_cost), 1.0)
    energy_scale = max(abs(e_batt * n_star), 1.0)
    g_values = Float64[]

    x_values = Vector{Float64}(undef, grid_size)
    x_values[1] = soc0 * n_star * e_batt
    for t in 1:n_t
        w_idx = 6 * (t - 1)
        u_idx = u_dim * (t - 1)
        x_values[t + 1] = x_values[t] + (
            w_scenario[w_idx + 5] * u_star[u_idx + 1] -
            w_scenario[w_idx + 6] * u_star[u_idx + 2]
        ) * delta_t
    end

    for t in 1:grid_size
        push!(g_values, (soc_min * n_star * e_batt - x_values[t]) / energy_scale)
        push!(g_values, (x_values[t] - soc_max * n_star * e_batt) / energy_scale)
    end

    for t in 1:n_t
        w_idx = 6 * (t - 1)
        u_idx = u_dim * (t - 1)
        required_grid_net = (
            w_scenario[w_idx + 1] -
            s_star * w_scenario[w_idx + 2] -
            u_star[u_idx + 2] +
            u_star[u_idx + 1]
        )
        push!(g_values, required_grid_net - pg_max)
        push!(g_values, -required_grid_net - pg_max)
    end

    for t in 1:n_t
        u_idx = u_dim * (t - 1)
        z_idx = 2 * (t - 1)
        l_idx = l_dim_per_t * (t - 1)
        push!(g_values, lambda_scenario[l_idx + 1] * u_star[u_idx + 1] + lambda_scenario[l_idx + 2] * u_star[u_idx + 2])
        push!(g_values, lambda_scenario[l_idx + 3] * z_scenario[z_idx + 1] + lambda_scenario[l_idx + 4] * z_scenario[z_idx + 2])
    end

    g_epi = (
        n_star * e_batt * battery_cost +
        s_star * base_pv_capacity_kw * pv_cost +
        sum(
            delta_t * (
                w_scenario[6 * (t - 1) + 3] * z_scenario[2 * (t - 1) + 1] -
                w_scenario[6 * (t - 1) + 4] * z_scenario[2 * (t - 1) + 2]
            )
            for t in 1:n_t
        ) - gamma_star
    ) / cost_scale
    push!(g_values, g_epi)

    return g_values
end

function _active_s_start(g_values; eps=1e-4)
    n = length(g_values)
    s_start = fill(eps / max(n - 1, 1), n)
    s_start[argmax(g_values)] = 1.0 - eps
    return s_start
end

include("add_master_z_scenario.jl")

function solve_local_red_z_lambda(
    max_z_model,
    z_initial, 
    lambda_initial,
    z_scenarios_history,
    lambda_scenarios_history,
    k1_iter,
    ipopt_lin_solver,
    max_iter_ipopt,
    grid_size,
    u_star,
    gamma_star,
    P,
    p,
    H,
    h,
    pc_max,
    pd_max,
    pg_max,
    n_max,
    soc0,
    soc_min,
    soc_max,
    n_star,
    s_star,
    battery_cost,
    pv_cost,
    base_pv_capacity_kw,
    delta_t,
    e_batt,
    k2_max,
    tol_2;
    is_silent=true,
    z_master_restarts::Int=1,
    z_master_restart_seed::Int=1234,
    z_master_recovery_restarts::Int=20,
    show_z_iter::Bool=false,
    show_w_vec::Bool=false,
    solve_stats=nothing,
    is_mip::Bool=false,
    mip_optimizer="Gurobi",
    is_cycle::Bool=false,
)
    """
    Run the middle-level local reduction algorithm to find the worst case violation w.
    The algorithm solves the master problem for a fixed number of y scenarios:

    max_{w in W, sigma} sigma
    s.t. forall k in {1,...,k2_max}:
        lambda_{1,k} * (sigma - (c'x_star + b'y_k - gamma_star))
            + sum_j (lambda_{j,k} *
            (sigma - (A_tilde*x_star + B_tilde*y_k + C_tilde*w - b_tilde)_j))<= 0

    This problem is a NLP because of the bilinear constraints.
    The maximization problem is solved by running a nested local reduction algorithm.
    """
    # Initialize the master z problem existence variables
    u_dim = 2
    nlp_logic_rows_per_t = 2
    milp_logic_rows_per_t = 8
    s_blocks = Vector{Vector{VariableRef}}(undef, k2_max)
    # Build master z problem (used to solve for z in the local reduction subroutine)
    ypad = "    "
    master_z, s_blocks = build_z_master(
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
        is_silent,
        is_cycle=is_cycle,
    )
    # Add first scenario constraint to master
    # Smoothed disjunction constraints for an initial scenario z/lambda
    w = master_z[:w]
    sigma = master_z[:sigma]
    n_t = grid_size - 1
    n_rows_s = master_z_selector_row_count(is_mip, grid_size)
    cost_scale = max(abs(base_pv_capacity_kw * pv_cost), abs(e_batt * battery_cost), 1.0)
    n_w = size(P, 2)
    w_ub = p[1:n_w]
    w_lb = -p[n_w + 1:2 * n_w]

    scenario_sigma_ubs = Float64[]
    add_master_z_scenario!(
        master_z,
        s_blocks,
        1,
        z_initial,
        lambda_initial,
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
        cost_scale,
        is_mip=is_mip,
    )

    # Initialze variables
    sigma_star = 0.0
    w_star = zeros(size(P, 2))
    z_star = copy(z_initial)
    z_scenarios_used = 1
    sigma_bar_star = 0.0
    x_witness = zeros(grid_size)
    w_mid = 0.5 .* (-p[size(P, 2) + 1:2 * size(P, 2)] .+ p[1:size(P, 2)])
    s_start_vectors = Union{Nothing, Vector{Float64}}[nothing for _ in 1:z_scenarios_used]
    if !is_mip
        s_start_vectors[1] = _active_s_start(
            _z_master_g_row_values(
                z_initial,
                lambda_initial,
                w_mid,
                u_star,
                gamma_star,
                n_star,
                s_star,
                soc0,
                soc_min,
                soc_max,
                pg_max,
                battery_cost,
                pv_cost,
                base_pv_capacity_kw,
                e_batt,
                delta_t=delta_t,
            ),
        )
    end
    initialize_master_z!(
        master_z,
        s_blocks,
        z_scenarios_used,
        P,
        p,
        u_star,
        soc0,
        n_star,
        delta_t,
        e_batt,
        z_witness=z_initial,
        sigma_start=sigma_star,
        s_start_vectors=s_start_vectors,
    )

    # Run the local reduction algorithm to find a violating scenario w (maximization step)
    z_iter = 0
    println(ypad, repeat("=", 20), " RUN LOCAL REDUCTION Z ", repeat("=", 20))
    while true
        z_iter += 1
        # Solve the master z problem for a fixed number of restarts to obtain a better local solution
        n_z_master_solves = is_mip ? 1 : z_master_restarts
        sigma_star, w_star, z_star, x_witness = solve_master_z_with_restarts!(
            master_z,
            s_blocks,
            z_scenarios_used,
            n_z_master_solves,
            z_master_restart_seed + z_iter - 1,
            is_silent,
            solve_stats=solve_stats,
            solve_label=is_mip ? "MIP" : "NLP",
            recovery_restarts=z_master_recovery_restarts,
        )

        if show_z_iter
            println(ypad, "Z iteration ", z_iter, " | obj(sigma)=", sigma_star, " | #z scenarios=", z_scenarios_used)
            if show_w_vec
                println(ypad, "  current w*=", w_star)
            end
        end


        # Set the current solutions to the master max y problem
        fix.(max_z_model[:u_star_var], u_star, force = true)
        fix.(max_z_model[:w_star_var], w_star, force = true)
        fix(max_z_model[:gamma_star_var], gamma_star, force = true)
        fix(max_z_model[:sigma_star_var], sigma_star, force = true)
        fix(max_z_model[:n_star_var], n_star, force = true)
        fix(max_z_model[:s_star_var], s_star, force = true)

        solve_t0 = time()
        optimize!(max_z_model)
        solve_elapsed = time() - solve_t0
        if solve_stats !== nothing
            if is_mip
                solve_stats[:mip_count] += 1
                solve_stats[:mip_time] += solve_elapsed
            else
                solve_stats[:nlp_count] += 1
                solve_stats[:nlp_time] += solve_elapsed
            end
        end

        max_z_ts = termination_status(max_z_model)
        max_z_ps = primal_status(max_z_model)
        if !has_values(max_z_model)
            println(ypad, "  Max-z solve returned no primal solution.")
            println(ypad, "  Termination status: ", max_z_ts)
            println(ypad, "  Primal status: ", max_z_ps)
            println(ypad, "  Stopping Z local reduction with the current z-master solution.")
            println(ypad)
            break
        end

        lambda_star = value.(max_z_model[:lambda])
        z_star = value.(max_z_model[:z])
        sigma_bar_star = objective_value(max_z_model)

        if show_z_iter
            println(ypad, "  sigma_bar*=", sigma_bar_star, " | max-z status=", max_z_ts)
        end

        # Check if the lambda local reduction found a violation above the tolerance level
        if sigma_bar_star > tol_2
            if z_scenarios_used + 1 > k2_max
                println(ypad, "Z-stop: reached max z scenarios.")
                println(ypad)
                break
            end
            add_master_z_scenario!(
                master_z,
                s_blocks,
                z_scenarios_used + 1,
                z_star,
                lambda_star,
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
                cost_scale,
                is_mip=is_mip,
            )
            
            z_scenarios_used += 1
            z_scenarios_history[:, z_scenarios_used, k1_iter] = z_star
            lambda_scenarios_history[:, z_scenarios_used, k1_iter] = lambda_star
            s_start_vectors = Union{Nothing, Vector{Float64}}[nothing for _ in 1:z_scenarios_used]
            if !is_mip
                s_start_vectors[z_scenarios_used] = _active_s_start(
                    _z_master_g_row_values(
                        z_star,
                        lambda_star,
                        w_star,
                        u_star,
                        gamma_star,
                        n_star,
                        s_star,
                        soc0,
                        soc_min,
                        soc_max,
                        pg_max,
                        battery_cost,
                        pv_cost,
                        base_pv_capacity_kw,
                        e_batt,
                        delta_t=delta_t,
                    ),
                )
            end
            initialize_master_z!(
                master_z,
                s_blocks,
                z_scenarios_used,
                P,
                p,
                u_star,
                soc0,
                n_star,
                delta_t,
                e_batt,
                w_start=w_star,
                x_witness=x_witness,
                z_witness=z_star,
                sigma_start=sigma_star,
                s_start_vectors=s_start_vectors,
            )
            println(ypad, "  Added Z scenario ", z_scenarios_used, " | new #z scenarios=", z_scenarios_used)
            println(ypad)
        else
            println(ypad, "Z-stop: no violation above tolerance.")
            println(ypad)
            break
        end
    end

    return sigma_star, w_star, x_witness, z_scenarios_used, z_scenarios_history, lambda_scenarios_history
end
