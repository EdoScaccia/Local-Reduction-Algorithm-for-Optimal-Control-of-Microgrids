include("build_z_master.jl")
include("build_z_master_milp.jl")
include("build_max_z_lp.jl")
include("assert_solution.jl")
include("solve_master_z_with_restarts.jl")
include("initialize_master_z.jl")
include("compute_milp_selector_big_m.jl")
if !(@isdefined local_reduction_g_row_count)
    include("local_reduction_state_rows.jl")
end

function _z_master_g_row_values(
    z_scenario,
    lambda_scenario,
    w_scenario,
    u_star,
    gamma_star,
    n_star,
    m_star,
    s_star,
    battery_cost,
    pv_cost,
    base_pv_capacity_kw,
    e_batt;
    soc0,
    soc_min,
    soc_max,
    delta_t,
    energy_scale,
    two_batt::Bool=true,
    is_cycle::Bool=false,
)
    u_dim = two_batt ? 4 : 2
    l_dim_per_t = two_batt ? 6 : 4
    n_t = div(length(u_star), u_dim)
    cost_scale = max(abs(base_pv_capacity_kw * pv_cost), abs(e_batt * battery_cost), 1.0)
    g_values = Float64[]
    x1 = Vector{Float64}(undef, n_t + 1)
    x2 = two_batt ? Vector{Float64}(undef, n_t + 1) : Float64[]
    x1[1] = soc0 * n_star * e_batt
    if two_batt
        x2[1] = soc0 * m_star * e_batt
    end

    for t in 1:n_t
        w_idx = 6 * (t - 1)
        u_idx = u_dim * (t - 1)
        z_idx = 2 * (t - 1)
        l_idx = l_dim_per_t * (t - 1)

        PL_t = w_scenario[w_idx + 1]
        PR_t = w_scenario[w_idx + 2]

        push!(g_values, lambda_scenario[l_idx + 1] * u_star[u_idx + 1] + lambda_scenario[l_idx + 2] * u_star[u_idx + 2])
        push!(g_values, lambda_scenario[l_idx + 3] * z_scenario[z_idx + 1] + lambda_scenario[l_idx + 4] * z_scenario[z_idx + 2])
        if two_batt
            push!(g_values, lambda_scenario[l_idx + 5] * u_star[u_idx + 3] + lambda_scenario[l_idx + 6] * u_star[u_idx + 4])
        end

        g_bal_plus_t = z_scenario[z_idx + 1] - z_scenario[z_idx + 2] + u_star[u_idx + 2] - u_star[u_idx + 1] + (two_batt ? u_star[u_idx + 4] - u_star[u_idx + 3] : 0.0) - PL_t + s_star * PR_t
        push!(g_values, g_bal_plus_t)
        push!(g_values, -g_bal_plus_t)

        eta_ch_t = w_scenario[w_idx + 5]
        rho_dis_t = w_scenario[w_idx + 6]
        x1[t + 1] = x1[t] + (eta_ch_t * u_star[u_idx + 1] - rho_dis_t * u_star[u_idx + 2]) * delta_t
        if two_batt
            x2[t + 1] = x2[t] + (eta_ch_t * u_star[u_idx + 3] - rho_dis_t * u_star[u_idx + 4]) * delta_t
        end
    end

    for t in 1:(n_t + 1)
        push!(g_values, (soc_min * n_star * e_batt - x1[t]) / energy_scale)
        push!(g_values, (x1[t] - soc_max * n_star * e_batt) / energy_scale)
        if two_batt
            push!(g_values, (soc_min * m_star * e_batt - x2[t]) / energy_scale)
            push!(g_values, (x2[t] - soc_max * m_star * e_batt) / energy_scale)
        end
    end

    if is_cycle
        terminal1 = (x1[end] - soc0 * n_star * e_batt) / energy_scale
        push!(g_values, terminal1)
        push!(g_values, -terminal1)
        if two_batt
            terminal2 = (x2[end] - soc0 * m_star * e_batt) / energy_scale
            push!(g_values, terminal2)
            push!(g_values, -terminal2)
        end
    end

    g_epi = delta_t * sum(
        w_scenario[6 * (t - 1) + 3] * z_scenario[2 * (t - 1) + 1] -
        w_scenario[6 * (t - 1) + 4] * z_scenario[2 * (t - 1) + 2]
        for t in 1:n_t
    ) + (n_star + (two_batt ? m_star : 0.0)) * e_batt * battery_cost + s_star * base_pv_capacity_kw * pv_cost - gamma_star
    g_epi /= cost_scale
    push!(g_values, g_epi)

    return g_values
end

function _active_s_start(g_values; eps=1e-4)
    n = length(g_values)
    s_start = fill(eps / max(n - 1, 1), n)
    s_start[argmax(g_values)] = 1.0 - eps
    return s_start
end

function _push_milp_row!(g_rows, g_lbs, g_ubs, expr, lb, ub)
    push!(g_rows, expr)
    push!(g_lbs, Float64(lb))
    push!(g_ubs, Float64(ub))
    return nothing
end

function _push_state_row!(g_rows, g_lbs, g_ubs, expr, lb, ub; collect_bounds::Bool)
    push!(g_rows, expr)
    if collect_bounds
        push!(g_lbs, Float64(lb))
        push!(g_ubs, Float64(ub))
    end
    return nothing
end

function _append_state_g_rows!(
    g_rows,
    g_lbs,
    g_ubs,
    x,
    state_bounds,
    grid_size,
    soc0,
    soc_min,
    soc_max,
    n_star,
    m_star,
    e_batt,
    energy_scale;
    two_batt::Bool=true,
    is_cycle::Bool=false,
    collect_bounds::Bool=false,
)
    cap1 = n_star * e_batt
    cap2 = two_batt ? m_star * e_batt : 0.0
    for t in 1:grid_size
        lower_expr = (soc_min * cap1 - x[t]) / energy_scale
        upper_expr = (x[t] - soc_max * cap1) / energy_scale
        if collect_bounds
            _push_state_row!(
                g_rows,
                g_lbs,
                g_ubs,
                lower_expr,
                (soc_min * cap1 - state_bounds.x1_max[t]) / energy_scale,
                (soc_min * cap1 - state_bounds.x1_min[t]) / energy_scale,
                collect_bounds=collect_bounds,
            )
            _push_state_row!(
                g_rows,
                g_lbs,
                g_ubs,
                upper_expr,
                (state_bounds.x1_min[t] - soc_max * cap1) / energy_scale,
                (state_bounds.x1_max[t] - soc_max * cap1) / energy_scale,
                collect_bounds=collect_bounds,
            )
        else
            _push_state_row!(g_rows, g_lbs, g_ubs, lower_expr, 0.0, 0.0, collect_bounds=collect_bounds)
            _push_state_row!(g_rows, g_lbs, g_ubs, upper_expr, 0.0, 0.0, collect_bounds=collect_bounds)
        end

        if two_batt
            lower_expr = (soc_min * cap2 - x[grid_size + t]) / energy_scale
            upper_expr = (x[grid_size + t] - soc_max * cap2) / energy_scale
            if collect_bounds
                _push_state_row!(
                    g_rows,
                    g_lbs,
                    g_ubs,
                    lower_expr,
                    (soc_min * cap2 - state_bounds.x2_max[t]) / energy_scale,
                    (soc_min * cap2 - state_bounds.x2_min[t]) / energy_scale,
                    collect_bounds=collect_bounds,
                )
                _push_state_row!(
                    g_rows,
                    g_lbs,
                    g_ubs,
                    upper_expr,
                    (state_bounds.x2_min[t] - soc_max * cap2) / energy_scale,
                    (state_bounds.x2_max[t] - soc_max * cap2) / energy_scale,
                    collect_bounds=collect_bounds,
                )
            else
                _push_state_row!(g_rows, g_lbs, g_ubs, lower_expr, 0.0, 0.0, collect_bounds=collect_bounds)
                _push_state_row!(g_rows, g_lbs, g_ubs, upper_expr, 0.0, 0.0, collect_bounds=collect_bounds)
            end
        end
    end

    if is_cycle
        terminal1 = (x[grid_size] - soc0 * cap1) / energy_scale
        if collect_bounds
            _push_state_row!(
                g_rows,
                g_lbs,
                g_ubs,
                terminal1,
                (state_bounds.x1_min[end] - soc0 * cap1) / energy_scale,
                (state_bounds.x1_max[end] - soc0 * cap1) / energy_scale,
                collect_bounds=collect_bounds,
            )
            _push_state_row!(
                g_rows,
                g_lbs,
                g_ubs,
                -terminal1,
                (soc0 * cap1 - state_bounds.x1_max[end]) / energy_scale,
                (soc0 * cap1 - state_bounds.x1_min[end]) / energy_scale,
                collect_bounds=collect_bounds,
            )
        else
            _push_state_row!(g_rows, g_lbs, g_ubs, terminal1, 0.0, 0.0, collect_bounds=collect_bounds)
            _push_state_row!(g_rows, g_lbs, g_ubs, -terminal1, 0.0, 0.0, collect_bounds=collect_bounds)
        end

        if two_batt
            terminal2 = (x[2 * grid_size] - soc0 * cap2) / energy_scale
            if collect_bounds
                _push_state_row!(
                    g_rows,
                    g_lbs,
                    g_ubs,
                    terminal2,
                    (state_bounds.x2_min[end] - soc0 * cap2) / energy_scale,
                    (state_bounds.x2_max[end] - soc0 * cap2) / energy_scale,
                    collect_bounds=collect_bounds,
                )
                _push_state_row!(
                    g_rows,
                    g_lbs,
                    g_ubs,
                    -terminal2,
                    (soc0 * cap2 - state_bounds.x2_max[end]) / energy_scale,
                    (soc0 * cap2 - state_bounds.x2_min[end]) / energy_scale,
                    collect_bounds=collect_bounds,
                )
            else
                _push_state_row!(g_rows, g_lbs, g_ubs, terminal2, 0.0, 0.0, collect_bounds=collect_bounds)
                _push_state_row!(g_rows, g_lbs, g_ubs, -terminal2, 0.0, 0.0, collect_bounds=collect_bounds)
            end
        end
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
    m_max,
    soc0,
    soc_min,
    soc_max,
    n_star,
    m_star,
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
    show_z_iter::Bool=false,
    show_w_vec::Bool=false,
    solve_stats=nothing,
    is_milp::Bool=false,
    milp_optimizer="HiGHS",
    two_batt::Bool=true,
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
    u_dim = two_batt ? 4 : 2
    nlp_l_dim_per_t = two_batt ? 6 : 4
    milp_l_dim_per_t = two_batt ? 3 : 2
    s_blocks = Vector{Vector{VariableRef}}(undef, k2_max)
    # Build master z problem (used to solve for z in the local reduction subroutine)
    ypad = "    "
    master_z, s_blocks = is_milp ?
        build_z_master_milp(
            s_blocks,
            milp_optimizer,
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
            is_silent,
            two_batt=two_batt,
            is_cycle=is_cycle,
        ) :
        build_z_master(
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
            is_silent,
            two_batt=two_batt,
            is_cycle=is_cycle,
        )
    # Add first scenario constraint to master
    # Smoothed disjunction constraints for an initial scenario z/lambda
    w = master_z[:w]
    sigma = master_z[:sigma]
    n_t = grid_size - 1
    n_rows_s = local_reduction_g_row_count(grid_size; two_batt=two_batt, is_milp=is_milp, is_cycle=is_cycle)
    cost_scale = max(abs(base_pv_capacity_kw * pv_cost), abs(e_batt * battery_cost), 1.0)
    energy_scale = max(abs(e_batt * max(n_star, two_batt ? m_star : n_star)), 1.0)
    n_w = size(P, 2)
    w_ub = p[1:n_w]
    w_lb = -p[n_w + 1:2 * n_w]
    state_bounds = fixed_dispatch_state_envelope(
        u_star,
        w_lb,
        w_ub,
        grid_size,
        soc0,
        n_star,
        m_star,
        delta_t,
        e_batt;
        two_batt=two_batt,
    )

    g_rows = Any[]
    g_lbs = Float64[]
    g_ubs = Float64[]
    scenario_sigma_ubs = Float64[]
    for t in 1:n_t
        w_idx = 6 * (t - 1)
        u_idx = u_dim * (t - 1)
        z_idx = 2 * (t - 1)
        PL_t = w[w_idx + 1, 1]
        PR_t = w[w_idx + 2, 1]

        if is_milp
            l_idx = milp_l_dim_per_t * (t - 1)
            _push_milp_row!(g_rows, g_lbs, g_ubs, -u_star[u_idx + 1], -u_star[u_idx + 1], -u_star[u_idx + 1])
            _push_milp_row!(g_rows, g_lbs, g_ubs, u_star[u_idx + 1] - n_max * pc_max * lambda_initial[l_idx + 1], u_star[u_idx + 1] - n_max * pc_max * lambda_initial[l_idx + 1], u_star[u_idx + 1] - n_max * pc_max * lambda_initial[l_idx + 1])
            _push_milp_row!(g_rows, g_lbs, g_ubs, -u_star[u_idx + 2], -u_star[u_idx + 2], -u_star[u_idx + 2])
            _push_milp_row!(g_rows, g_lbs, g_ubs, u_star[u_idx + 2] - n_max * pd_max * (1 - lambda_initial[l_idx + 1]), u_star[u_idx + 2] - n_max * pd_max * (1 - lambda_initial[l_idx + 1]), u_star[u_idx + 2] - n_max * pd_max * (1 - lambda_initial[l_idx + 1]))
            if two_batt
                _push_milp_row!(g_rows, g_lbs, g_ubs, -u_star[u_idx + 3], -u_star[u_idx + 3], -u_star[u_idx + 3])
                _push_milp_row!(g_rows, g_lbs, g_ubs, u_star[u_idx + 3] - m_max * pc_max * lambda_initial[l_idx + 2], u_star[u_idx + 3] - m_max * pc_max * lambda_initial[l_idx + 2], u_star[u_idx + 3] - m_max * pc_max * lambda_initial[l_idx + 2])
                _push_milp_row!(g_rows, g_lbs, g_ubs, -u_star[u_idx + 4], -u_star[u_idx + 4], -u_star[u_idx + 4])
                _push_milp_row!(g_rows, g_lbs, g_ubs, u_star[u_idx + 4] - m_max * pd_max * (1 - lambda_initial[l_idx + 2]), u_star[u_idx + 4] - m_max * pd_max * (1 - lambda_initial[l_idx + 2]), u_star[u_idx + 4] - m_max * pd_max * (1 - lambda_initial[l_idx + 2]))
            end
            grid_l = l_idx + (two_batt ? 3 : 2)
            _push_milp_row!(g_rows, g_lbs, g_ubs, -z_initial[z_idx + 1], -z_initial[z_idx + 1], -z_initial[z_idx + 1])
            _push_milp_row!(g_rows, g_lbs, g_ubs, z_initial[z_idx + 1] - pg_max * lambda_initial[grid_l], z_initial[z_idx + 1] - pg_max * lambda_initial[grid_l], z_initial[z_idx + 1] - pg_max * lambda_initial[grid_l])
            _push_milp_row!(g_rows, g_lbs, g_ubs, -z_initial[z_idx + 2], -z_initial[z_idx + 2], -z_initial[z_idx + 2])
            _push_milp_row!(g_rows, g_lbs, g_ubs, z_initial[z_idx + 2] - pg_max * (1 - lambda_initial[grid_l]), z_initial[z_idx + 2] - pg_max * (1 - lambda_initial[grid_l]), z_initial[z_idx + 2] - pg_max * (1 - lambda_initial[grid_l]))
        else
            l_idx = nlp_l_dim_per_t * (t - 1)
            # Logic rows
            push!(g_rows, lambda_initial[l_idx + 1] * u_star[u_idx + 1] + lambda_initial[l_idx + 2] * u_star[u_idx + 2])
            push!(g_rows, lambda_initial[l_idx + 3] * z_initial[z_idx + 1] + lambda_initial[l_idx + 4] * z_initial[z_idx + 2])
            if two_batt
                push!(g_rows, lambda_initial[l_idx + 5] * u_star[u_idx + 3] + lambda_initial[l_idx + 6] * u_star[u_idx + 4])
            end
        end

        # Load-balance equality split into two inequalities
        g_bal_plus_t = z_initial[z_idx + 1] - z_initial[z_idx + 2] + u_star[u_idx + 2] - u_star[u_idx + 1] + (two_batt ? u_star[u_idx + 4] - u_star[u_idx + 3] : 0.0) - PL_t + s_star * PR_t
        push!(g_rows, g_bal_plus_t)
        push!(g_rows, -g_bal_plus_t)
        if is_milp
            bal_const = z_initial[z_idx + 1] - z_initial[z_idx + 2] + u_star[u_idx + 2] - u_star[u_idx + 1] + (two_batt ? u_star[u_idx + 4] - u_star[u_idx + 3] : 0.0)
            bal_lb = bal_const - w_ub[w_idx + 1] + s_star * w_lb[w_idx + 2]
            bal_ub = bal_const - w_lb[w_idx + 1] + s_star * w_ub[w_idx + 2]
            append!(g_lbs, [bal_lb, -bal_ub])
            append!(g_ubs, [bal_ub, -bal_lb])
        end
    end

    _append_state_g_rows!(
        g_rows,
        g_lbs,
        g_ubs,
        master_z[:x],
        state_bounds,
        grid_size,
        soc0,
        soc_min,
        soc_max,
        n_star,
        m_star,
        e_batt,
        energy_scale;
        two_batt=two_batt,
        is_cycle=is_cycle,
        collect_bounds=is_milp,
    )

    # Epigraph row
    g_epi = delta_t * sum(
        w[6 * (t - 1) + 3, 1] * z_initial[2 * (t - 1) + 1] -
        w[6 * (t - 1) + 4, 1] * z_initial[2 * (t - 1) + 2]
        for t in 1:n_t
    ) + (n_star + (two_batt ? m_star : 0.0)) * e_batt * battery_cost + s_star * base_pv_capacity_kw * pv_cost - gamma_star
    g_epi /= cost_scale
    push!(g_rows, g_epi)
    if is_milp
        epi_fixed = (n_star + (two_batt ? m_star : 0.0)) * e_batt * battery_cost + s_star * base_pv_capacity_kw * pv_cost - gamma_star
        epi_lb = (
                delta_t * sum(
                    w_lb[6 * (t - 1) + 3] * z_initial[2 * (t - 1) + 1] -
                    w_ub[6 * (t - 1) + 4] * z_initial[2 * (t - 1) + 2]
                    for t in 1:n_t
            ) + epi_fixed
        ) / cost_scale
        epi_ub = (
                delta_t * sum(
                    w_ub[6 * (t - 1) + 3] * z_initial[2 * (t - 1) + 1] -
                    w_lb[6 * (t - 1) + 4] * z_initial[2 * (t - 1) + 2]
                    for t in 1:n_t
            ) + epi_fixed
        ) / cost_scale
        push!(g_lbs, epi_lb)
        push!(g_ubs, epi_ub)
    end

    if length(g_rows) != n_rows_s
        error("Mismatch in number of g_i rows: expected $n_rows_s, got $(length(g_rows)).")
    end

    if is_milp
        push!(scenario_sigma_ubs, milp_selector_scenario_sigma_ub(g_ubs))
        sigma_ub = milp_selector_global_sigma_ub(scenario_sigma_ubs)
        _add_milp_selector_constraints!(master_z, sigma, s_blocks[1], g_rows, g_lbs, sigma_ub)
    else
        @constraint(
            master_z,
            sum(s_blocks[1][i] * (sigma - g_rows[i]) for i in 1:n_rows_s) <= 0,
        )
    end

    # Initialze variables
    sigma_star = 0.0
    w_star = zeros(size(P, 2))
    z_scenarios_used = 1
    sigma_bar_star = 0.0
    x_witness = zeros((two_batt ? 2 : 1) * grid_size)
    z_termination_reason = "not_started"
    w_mid = 0.5 .* (-p[size(P, 2) + 1:2 * size(P, 2)] .+ p[1:size(P, 2)])
    s_start_vectors = Union{Nothing, Vector{Float64}}[nothing for _ in 1:z_scenarios_used]
    if !is_milp
        s_start_vectors[1] = _active_s_start(
            _z_master_g_row_values(
                z_initial,
                lambda_initial,
                w_mid,
                u_star,
                gamma_star,
                n_star,
                m_star,
                s_star,
                battery_cost,
                pv_cost,
                base_pv_capacity_kw,
                e_batt,
                soc0=soc0,
                soc_min=soc_min,
                soc_max=soc_max,
                delta_t=delta_t,
                energy_scale=energy_scale,
                two_batt=two_batt,
                is_cycle=is_cycle,
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
        m_star,
        delta_t,
        e_batt,
        sigma_start=sigma_star,
        s_start_vectors=s_start_vectors,
        two_batt=two_batt,
    )

    # Run the local reduction algorithm to find a violating scenario w (maximization step)
    z_iter = 0
    println(ypad, repeat("=", 20), " RUN LOCAL REDUCTION Z ", repeat("=", 20))
    while true
        z_iter += 1
        # Solve the master z problem for a fixed number of restarts to obtain a better local solution
        n_z_master_solves = is_milp ? 1 : z_master_restarts
        sigma_star, w_star, x_witness = solve_master_z_with_restarts!(
            master_z,
            s_blocks,
            z_scenarios_used,
            n_z_master_solves,
            z_master_restart_seed + z_iter - 1,
            is_silent,
            solve_stats=solve_stats,
            solve_label=is_milp ? "MILP" : "NLP",
        )

        if show_z_iter
            println(ypad, "Z iteration ", z_iter, " | obj(sigma)=", sigma_star, " | #z scenarios=", z_scenarios_used)
            if show_w_vec
                println(ypad, "  current w*=", w_star)
            end
        end

        if is_milp
            max_z_model = build_max_z_lp_fixed(
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
                is_silent,
                two_batt=two_batt,
                is_cycle=is_cycle,
            )
        else
            # Set the current solutions to the master max y problem
            fix.(max_z_model[:u_star_var], u_star, force = true)
            fix.(max_z_model[:w_star_var], w_star, force = true)
            fix(max_z_model[:gamma_star_var], gamma_star, force = true)
            fix(max_z_model[:sigma_star_var], sigma_star, force = true)
            fix(max_z_model[:energy_scale_star_var], energy_scale, force = true)
            fix(max_z_model[:n_star_var], n_star, force = true)
            if two_batt
                fix(max_z_model[:m_star_var], m_star, force = true)
            end
            fix(max_z_model[:s_star_var], s_star, force = true)
        end
        solve_t0 = time()
        optimize!(max_z_model)
        solve_elapsed = time() - solve_t0
        if solve_stats !== nothing
            solve_stats[:nlp_count] += 1
            solve_stats[:nlp_time] += solve_elapsed
        end
        z_star = value.(max_z_model[:z])
        lambda_star = value.(max_z_model[:lambda])
        sigma_bar_star = objective_value(max_z_model)

        if show_z_iter
            println(ypad, "  sigma_bar*=", sigma_bar_star)
        end

        # Check if the lambda local reduction found a violation above the tolerance level
        if sigma_bar_star > tol_2
            if z_scenarios_used + 1 > k2_max
                z_termination_reason = "max_z_scenarios_reached"
                println(ypad, "Z-stop: reached max z scenarios.")
                println(ypad)
                break
            end
            # Add new z scenario to the master z problem
            if is_milp
                s_blocks[z_scenarios_used + 1] = @variable(master_z, [1:n_rows_s], Bin)
                @constraint(master_z, sum(s_blocks[z_scenarios_used + 1]) == 1)
            else
                s_blocks[z_scenarios_used + 1] = @variable(master_z, [1:n_rows_s], lower_bound = 0.0)
                @constraint(master_z, sum(s_blocks[z_scenarios_used + 1]) == 1)
            end

            g_rows = Any[]
            g_lbs = Float64[]
            g_ubs = Float64[]
            for t in 1:n_t
                w_idx = 6 * (t - 1)
                u_idx = u_dim * (t - 1)
                z_idx = 2 * (t - 1)
                PL_t = w[w_idx + 1, 1]
                PR_t = w[w_idx + 2, 1]

                if is_milp
                    l_idx = milp_l_dim_per_t * (t - 1)
                    _push_milp_row!(g_rows, g_lbs, g_ubs, -u_star[u_idx + 1], -u_star[u_idx + 1], -u_star[u_idx + 1])
                    _push_milp_row!(g_rows, g_lbs, g_ubs, u_star[u_idx + 1] - n_max * pc_max * lambda_star[l_idx + 1], u_star[u_idx + 1] - n_max * pc_max * lambda_star[l_idx + 1], u_star[u_idx + 1] - n_max * pc_max * lambda_star[l_idx + 1])
                    _push_milp_row!(g_rows, g_lbs, g_ubs, -u_star[u_idx + 2], -u_star[u_idx + 2], -u_star[u_idx + 2])
                    _push_milp_row!(g_rows, g_lbs, g_ubs, u_star[u_idx + 2] - n_max * pd_max * (1 - lambda_star[l_idx + 1]), u_star[u_idx + 2] - n_max * pd_max * (1 - lambda_star[l_idx + 1]), u_star[u_idx + 2] - n_max * pd_max * (1 - lambda_star[l_idx + 1]))
                    if two_batt
                        _push_milp_row!(g_rows, g_lbs, g_ubs, -u_star[u_idx + 3], -u_star[u_idx + 3], -u_star[u_idx + 3])
                        _push_milp_row!(g_rows, g_lbs, g_ubs, u_star[u_idx + 3] - m_max * pc_max * lambda_star[l_idx + 2], u_star[u_idx + 3] - m_max * pc_max * lambda_star[l_idx + 2], u_star[u_idx + 3] - m_max * pc_max * lambda_star[l_idx + 2])
                        _push_milp_row!(g_rows, g_lbs, g_ubs, -u_star[u_idx + 4], -u_star[u_idx + 4], -u_star[u_idx + 4])
                        _push_milp_row!(g_rows, g_lbs, g_ubs, u_star[u_idx + 4] - m_max * pd_max * (1 - lambda_star[l_idx + 2]), u_star[u_idx + 4] - m_max * pd_max * (1 - lambda_star[l_idx + 2]), u_star[u_idx + 4] - m_max * pd_max * (1 - lambda_star[l_idx + 2]))
                    end
                    grid_l = l_idx + (two_batt ? 3 : 2)
                    _push_milp_row!(g_rows, g_lbs, g_ubs, -z_star[z_idx + 1], -z_star[z_idx + 1], -z_star[z_idx + 1])
                    _push_milp_row!(g_rows, g_lbs, g_ubs, z_star[z_idx + 1] - pg_max * lambda_star[grid_l], z_star[z_idx + 1] - pg_max * lambda_star[grid_l], z_star[z_idx + 1] - pg_max * lambda_star[grid_l])
                    _push_milp_row!(g_rows, g_lbs, g_ubs, -z_star[z_idx + 2], -z_star[z_idx + 2], -z_star[z_idx + 2])
                    _push_milp_row!(g_rows, g_lbs, g_ubs, z_star[z_idx + 2] - pg_max * (1 - lambda_star[grid_l]), z_star[z_idx + 2] - pg_max * (1 - lambda_star[grid_l]), z_star[z_idx + 2] - pg_max * (1 - lambda_star[grid_l]))
                else
                    l_idx = nlp_l_dim_per_t * (t - 1)
                    # Logic rows
                    push!(g_rows, lambda_star[l_idx + 1] * u_star[u_idx + 1] + lambda_star[l_idx + 2] * u_star[u_idx + 2])
                    push!(g_rows, lambda_star[l_idx + 3] * z_star[z_idx + 1] + lambda_star[l_idx + 4] * z_star[z_idx + 2])
                    if two_batt
                        push!(g_rows, lambda_star[l_idx + 5] * u_star[u_idx + 3] + lambda_star[l_idx + 6] * u_star[u_idx + 4])
                    end
                end

                # Load-balance equality split into two inequalities
                g_bal_plus_t = z_star[z_idx + 1] - z_star[z_idx + 2] + u_star[u_idx + 2] - u_star[u_idx + 1] + (two_batt ? u_star[u_idx + 4] - u_star[u_idx + 3] : 0.0) - PL_t + s_star * PR_t
                push!(g_rows, g_bal_plus_t)
                push!(g_rows, -g_bal_plus_t)
                if is_milp
                    bal_const = z_star[z_idx + 1] - z_star[z_idx + 2] + u_star[u_idx + 2] - u_star[u_idx + 1] + (two_batt ? u_star[u_idx + 4] - u_star[u_idx + 3] : 0.0)
                    bal_lb = bal_const - w_ub[w_idx + 1] + s_star * w_lb[w_idx + 2]
                    bal_ub = bal_const - w_lb[w_idx + 1] + s_star * w_ub[w_idx + 2]
                    append!(g_lbs, [bal_lb, -bal_ub])
                    append!(g_ubs, [bal_ub, -bal_lb])
                end
            end

            _append_state_g_rows!(
                g_rows,
                g_lbs,
                g_ubs,
                master_z[:x],
                state_bounds,
                grid_size,
                soc0,
                soc_min,
                soc_max,
                n_star,
                m_star,
                e_batt,
                energy_scale;
                two_batt=two_batt,
                is_cycle=is_cycle,
                collect_bounds=is_milp,
            )

            # Epigraph row
            g_epi = delta_t * sum(
                w[6 * (t - 1) + 3, 1] * z_star[2 * (t - 1) + 1] -
                w[6 * (t - 1) + 4, 1] * z_star[2 * (t - 1) + 2]
                for t in 1:n_t
            ) + (n_star + (two_batt ? m_star : 0.0)) * e_batt * battery_cost + s_star * base_pv_capacity_kw * pv_cost - gamma_star
            g_epi /= cost_scale
            push!(g_rows, g_epi)
            if is_milp
                epi_fixed = (n_star + (two_batt ? m_star : 0.0)) * e_batt * battery_cost + s_star * base_pv_capacity_kw * pv_cost - gamma_star
                epi_lb = (
                    delta_t * sum(
                        w_lb[6 * (t - 1) + 3] * z_star[2 * (t - 1) + 1] -
                        w_ub[6 * (t - 1) + 4] * z_star[2 * (t - 1) + 2]
                        for t in 1:n_t
                    ) + epi_fixed
                ) / cost_scale
                epi_ub = (
                    delta_t * sum(
                        w_ub[6 * (t - 1) + 3] * z_star[2 * (t - 1) + 1] -
                        w_lb[6 * (t - 1) + 4] * z_star[2 * (t - 1) + 2]
                        for t in 1:n_t
                    ) + epi_fixed
                ) / cost_scale
                push!(g_lbs, epi_lb)
                push!(g_ubs, epi_ub)
            end

            if length(g_rows) != n_rows_s
                error("Mismatch in number of g_i rows: expected $n_rows_s, got $(length(g_rows)).")
            end

            if is_milp
                push!(scenario_sigma_ubs, milp_selector_scenario_sigma_ub(g_ubs))
                sigma_ub = milp_selector_global_sigma_ub(scenario_sigma_ubs)
                _add_milp_selector_constraints!(master_z, sigma, s_blocks[z_scenarios_used + 1], g_rows, g_lbs, sigma_ub)
            else
                @constraint(
                    master_z,
                    sum(s_blocks[z_scenarios_used + 1][i] * (sigma - g_rows[i]) for i in 1:n_rows_s) <= 0,
                )
            end
            
            z_scenarios_history[:,z_iter, k1_iter] = z_star
            z_scenarios_used += 1
            lambda_scenarios_history[:, z_scenarios_used, k1_iter] = lambda_star
            s_start_vectors = Union{Nothing, Vector{Float64}}[nothing for _ in 1:z_scenarios_used]
            if !is_milp
                s_start_vectors[z_scenarios_used] = _active_s_start(
                    _z_master_g_row_values(
                        z_star,
                        lambda_star,
                        w_star,
                        u_star,
                        gamma_star,
                        n_star,
                        m_star,
                        s_star,
                        battery_cost,
                        pv_cost,
                        base_pv_capacity_kw,
                        e_batt,
                        soc0=soc0,
                        soc_min=soc_min,
                        soc_max=soc_max,
                        delta_t=delta_t,
                        energy_scale=energy_scale,
                        two_batt=two_batt,
                        is_cycle=is_cycle,
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
                m_star,
                delta_t,
                e_batt,
                w_start=w_star,
                x_witness=x_witness,
                sigma_start=sigma_star,
                s_start_vectors=s_start_vectors,
                two_batt=two_batt,
            )
            println(ypad, "  Added Z scenario ", z_scenarios_used, " | new #z scenarios=", z_scenarios_used)
            println(ypad)
        else
            z_termination_reason = "no_z_violation"
            println(ypad, "Z-stop: no violation above tolerance.")
            println(ypad)
            break
        end
    end

    return sigma_star, w_star, x_witness, z_scenarios_used, z_scenarios_history, lambda_scenarios_history, z_termination_reason
end
