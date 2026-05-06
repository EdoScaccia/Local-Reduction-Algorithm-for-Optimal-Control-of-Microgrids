using JuMP
using LinearAlgebra
using Dates
using JLD2

include("load_problem_data.jl")
include("print_initial_summary.jl")
include("print_final_summary.jl")
include("build_master_problem.jl")
include("build_master_problem_milp.jl")
include("find_initial_feasible_scenario_master.jl")
include("find_initial_feasible_scenario_z_lambda.jl")
include("build_max_z.jl")
include("solve_local_red_z_lambda.jl")
include("initialize_new_master_scenario.jl")
include("add_master_scenario.jl")
include("build_grid_feasibility_oracle.jl")
include("build_grid_feasibility_selector_oracle.jl")

function _run_config_file_stem(soc0, horizon_days, grid_power_factor)
    soc0_pct = round(Int, 100 * soc0)
    pg_pct = round(Int, 100 * grid_power_factor)
    return "SOC0$(soc0_pct)_H$(horizon_days)_PG$(pg_pct)"
end

function _last_finite_column(A)
    for j in axes(A, 2)[end]:-1:axes(A, 2)[begin]
        if all(isfinite, A[:, j])
            return j
        end
    end
    return nothing
end

function _last_finite_value(v)
    for i in eachindex(v)
        j = lastindex(v) - i + firstindex(v)
        isfinite(v[j]) && return v[j]
    end
    return nothing
end

function _find_latest_1day_result(results_dir, soc0, grid_power_factor)
    isdir(results_dir) || return nothing

    config_stem = _run_config_file_stem(soc0, 1, grid_power_factor)
    candidates = filter(
        path -> endswith(path, ".jld2") && occursin(config_stem, basename(path)),
        readdir(results_dir; join=true),
    )
    isempty(candidates) && return nothing

    return first(sort(candidates; by=mtime, rev=true))
end

function _repeat_control_to_horizon(u_day, n_t_target, u_dim)
    length(u_day) % u_dim == 0 || error("Warm-start control vector is incompatible with u_dim=$u_dim.")
    n_t_day = div(length(u_day), u_dim)
    n_t_day > 0 || error("Cannot warm start from an empty one-day control vector.")

    u_start = zeros(u_dim * n_t_target)
    for t in 1:n_t_target
        src_t = mod1(t, n_t_day)
        src_idx = u_dim * (src_t - 1)
        dst_idx = u_dim * (t - 1)
        u_start[dst_idx + 1:dst_idx + u_dim] .= u_day[src_idx + 1:src_idx + u_dim]
    end
    return u_start
end

function _load_1day_warm_start(
    results_dir,
    soc0,
    grid_power_factor,
    n_t_target;
    warm_start_file=nothing,
    u_dim=4,
    two_batt=true,
)
    result_path = warm_start_file === nothing ?
        _find_latest_1day_result(results_dir, soc0, grid_power_factor) :
        (isabspath(String(warm_start_file)) ? String(warm_start_file) : joinpath(results_dir, String(warm_start_file)))

    result_path === nothing && return nothing
    isfile(result_path) || return nothing

    data = JLD2.load(result_path)
    haskey(data, "u_star_history") || return nothing
    haskey(data, "m_star") || return nothing
    if haskey(data, "two_batt") && data["two_batt"] != two_batt
        return nothing
    elseif !haskey(data, "two_batt") && !two_batt
        return nothing
    end

    u_hist = data["u_star_history"]
    col = _last_finite_column(u_hist)
    col === nothing && return nothing

    u_day = u_hist[:, col]
    gamma_start = haskey(data, "master_x_obj_history") ? _last_finite_value(data["master_x_obj_history"]) : nothing
    n_start = haskey(data, "n_star") ? data["n_star"] : 1.0
    m_start = haskey(data, "m_star") ? data["m_star"] : n_start
    s_start = haskey(data, "s_star") ? data["s_star"] : 1.0

    return (
        result_path=result_path,
        u_start=_repeat_control_to_horizon(u_day, n_t_target, u_dim),
        gamma_start=gamma_start,
        n_start=n_start,
        m_start=m_start,
        s_start=s_start,
    )
end

function _save_local_reduction_results(
    results_dir::AbstractString,
    resolved_data_path,
    source_label,
    soc0,
    horizon_days,
    grid_power_factor,
    is_milp,
    two_batt,
    is_cycle,
    master_x_obj_history,
    u_star_history,
    w_scenarios_history,
    z_scenarios_history,
    lambda_scenarios_history,
    scenarios_added,
    solve_stats,
    iteration_log,
    master_x_solve_time_history,
    total_elapsed_seconds,
    n_star,
    m_star,
    s_star,
    uncertainty_bounds,
    microgrid_params,
    data_config,
)
    mkpath(results_dir)
    timestamp = Dates.format(now(), "yyyy-mm-dd_HHMMSS")
    config_stem = _run_config_file_stem(soc0, horizon_days, grid_power_factor)
    formulation_stem = is_milp ? "MILP" : "NLP"
    battery_stem = two_batt ? "B2" : "B1"
    stem = formulation_stem * "_" * config_stem * "_" * battery_stem
    results_path = joinpath(results_dir, stem * "_" * timestamp * ".jld2")
    JLD2.jldsave(
        results_path;
        resolved_data_path=resolved_data_path === nothing ? "" : String(resolved_data_path),
        data_source=String(source_label),
        saved_at=string(now()),
        is_milp=is_milp,
        two_batt=two_batt,
        is_cycle=is_cycle,
        master_x_obj_history=master_x_obj_history,
        u_star_history=u_star_history,
        w_scenarios_history=w_scenarios_history,
        z_scenarios_history=z_scenarios_history,
        lambda_scenarios_history=lambda_scenarios_history,
        scenarios_added=scenarios_added,
        solve_stats=solve_stats,
        iteration_log=iteration_log,
        master_x_solve_time_history=master_x_solve_time_history,
        total_elapsed_seconds=total_elapsed_seconds,
        n_star=n_star,
        m_star=m_star,
        s_star=s_star,
        uncertainty_bounds=uncertainty_bounds,
        microgrid_params=microgrid_params,
        data_config=data_config,
    )
    return results_path
end

function run_local_reduction_nonlinear_existence(
    problem_source=nothing;
    processed_data_path::AbstractString=joinpath(@__DIR__, "Processed_Data_15min.csv"),
    horizon_days::Int=1,
    initial_day="2016-05-01",
    soc0::Float64=0.4,
    load_error_pct::Float64=0.10,
    pv_error_pct::Float64=0.10,
    pv_factor::Float64=1.0,
    grid_power_factor::Float64=0.2,
    battery_factor::Float64=1.0,
    battery_cost::Float64=200.0,
    pv_cost::Float64=1500.0,
    base_pv_capacity_kw::Float64=90.84,
    k1_max::Int=5,
    k2_max::Int=15,
    k3_max::Int=20,
    ipopt_lin_solver="ma57",
    milp_optimizer="HiGHS",
    is_milp::Bool=false,
    max_iter_ipopt=5000,
    tol_1::Float64=1e-6,
    tol_2::Float64=1e-6,
    z_master_restarts::Int=5,
    z_master_restart_seed::Int=1234,
    show_w_vec::Bool=false,
    show_z_iter::Bool=true,
    is_silent::Bool=false,
    save_results::Bool=false,
    results_dir::AbstractString=joinpath(@__DIR__, "Results"),
    warm_start_from_1day::Bool=false,
    warm_start_1day_file=nothing,
    two_batt::Bool=true,
    is_cycle::Bool=false,
    use_grid_selector_oracle::Bool=false,
    n_max::Real=100,
    m_max::Real=100,
    s_max::Real=100,
)
    """
    Three-level local reduction scaffold for problems with nonlinear existence constraints.

    The expected problem data are loaded by `load_problem_data(problem_source)` and should
    provide:
    - `problem_name`
    - `master`: data required by the outer master
    - `uncertainty`: data required to sample/describe `w in W`
    - `recourse`: data required to sample/describe `y in Y`
    - `nonlinear_context`: any extra data required by the nonlinear existence callbacks
    """
    if min(k1_max, k2_max, k3_max) < 2
        error("k1_max, k2_max, and k3_max must all be at least 2.")
    end
    if z_master_restarts < 1
        error("z_master_restarts must be at least 1.")
    end
    if horizon_days < 1
        error("`horizon_days` must be at least 1.")
    end
    if load_error_pct < 0 || pv_error_pct < 0
        error("Prediction error percentages must be nonnegative.")
    end
    total_t0 = time()

    # Load the problem data (check if the test data are being requested)
    println("====================================================")
    println("Loading problem data...")
    flush(stdout)
    problem_data, uncertainty_bounds, uncertainty_set,
    control_set, grid_set, state_set, microgrid_params = load_problem_data(
        problem_source;
        processed_data_path=processed_data_path,
        horizon_days=horizon_days,
        initial_day=initial_day,
        soc0=soc0,
        load_error_pct=load_error_pct,
        pv_error_pct=pv_error_pct,
        pv_factor=pv_factor,
        grid_power_factor=grid_power_factor,
        battery_factor=battery_factor,
        battery_cost=battery_cost
    )

    P = uncertainty_set["P"]
    p = uncertainty_set["p"]
    n_w = uncertainty_set["n_w"]
    H = grid_set["H"]
    h = grid_set["h"]
    n_z = grid_set["n_z"]
    grid_size = microgrid_params["grid_size"]
    n_t = grid_size - 1
    u_dim = two_batt ? 4 : 2
    x_dim_factor = two_batt ? 2 : 1
    n_u = u_dim * n_t
    n_x = x_dim_factor * grid_size
    soc0 = microgrid_params["SoC0"]
    soc_min = microgrid_params["SOC_MIN"]
    soc_max = microgrid_params["SOC_MAX"]
    pc_max = microgrid_params["Pc_MAX"]
    pd_max = microgrid_params["Pd_MAX"]
    pg_max = microgrid_params["PG_plus_MAX"]
    delta_t = microgrid_params["Δt"]
    e_batt = microgrid_params["E_BATT"]
    cost_scale = max(abs(base_pv_capacity_kw * pv_cost), abs(e_batt * battery_cost), 1.0)
    energy_scale = max(abs(e_batt * max(n_max, m_max)), 1.0)
    println("Done.")

    k1_slots = k1_max

    print_initial_summary(
        n_x,
        n_u,
        n_z,
        n_w,
        k1_max,
        k2_max,
        k3_max,
        tol_1,
        tol_2,
        processed_data_path,
        horizon_days,
        initial_day,
        soc0,
        load_error_pct,
        pv_error_pct,
        pv_factor,
        grid_power_factor,
        battery_factor,
        battery_cost,
        pv_cost,
        base_pv_capacity_kw
    )

    # Initialize the master x problem existence variables
    x_blocks = Vector{Vector{VariableRef}}(undef, k1_max)
    z_blocks = Vector{Vector{VariableRef}}(undef, k1_max)
    l_blocks = Vector{Vector{VariableRef}}(undef, k1_max)

    # Find an initial feasible scenario for the master problem
    w_init = find_initial_feasible_scenario_master(P, p)

    # Build the master problem with the initial scenario
    if is_milp
        master_x, x_blocks, z_blocks, l_blocks = build_master_problem_milp(
            milp_optimizer,
            x_blocks,
            z_blocks,
            l_blocks,
            P,
            p,
            H,
            h,
            grid_size,
            soc0,
            soc_min,
            soc_max,
            pc_max,
            pd_max,
            pg_max,
            battery_cost,
            pv_cost,
            base_pv_capacity_kw,
            delta_t,
            e_batt,
            is_silent;
            w_init=w_init,
            two_batt=two_batt,
            is_cycle=is_cycle,
            n_max=n_max,
            m_max=m_max,
            s_max=s_max,
        )
    else
        master_x, x_blocks, z_blocks, l_blocks = build_master_problem(
            ipopt_lin_solver,
            max_iter_ipopt,
            x_blocks,
            z_blocks,
            l_blocks,
            P,
            p,
            H,
            h,
            grid_size,
            soc0,
            soc_min,
            soc_max,
            pc_max,
            pd_max,
            battery_cost,
            pv_cost,
            base_pv_capacity_kw,
            delta_t,
            e_batt,
            is_silent;
            w_init=w_init,
            two_batt=two_batt,
            is_cycle=is_cycle,
            n_max=n_max,
            m_max=m_max,
            s_max=s_max,
        )
    end
    u = master_x[:u]
    gamma = master_x[:gamma]
    n = master_x[:n]
    soc0_var = master_x[:SOC0]
    s = master_x[:s]

    if warm_start_from_1day && horizon_days > 1
        warm_start = _load_1day_warm_start(
            results_dir,
            soc0,
            grid_power_factor,
            n_t;
            warm_start_file=warm_start_1day_file,
            u_dim=u_dim,
            two_batt=two_batt,
        )

        if warm_start === nothing
            println("No matching 1-day warm-start result found. Using default master initialization.")
        else
            gamma_start = warm_start.gamma_start === nothing ? cost_scale : warm_start.gamma_start
            initialize_new_master_scenario!(
                master_x,
                x_blocks[1],
                z_blocks[1],
                l_blocks[1],
                warm_start.u_start,
                w_init,
                gamma_start,
                warm_start.n_start,
                warm_start.m_start,
                warm_start.s_start,
                soc0,
                delta_t,
                e_batt,
                two_batt=two_batt,
                is_milp=is_milp,
            )
            println("Warm-started master from 1-day result: ", warm_start.result_path)
        end
    end

    # Build the z maximization problem
    max_z_model = is_milp ?
        nothing :
        build_max_z(ipopt_lin_solver, max_iter_ipopt, H, h, grid_size, battery_cost, pv_cost, base_pv_capacity_kw, delta_t, e_batt, is_silent; two_batt=two_batt)
    grid_oracle_models = use_grid_selector_oracle ? nothing :
        build_grid_feasibility_oracle_models(
            P,
            p,
            grid_size,
            u_dim,
            milp_optimizer=milp_optimizer,
            is_silent=is_silent,
            two_batt=two_batt,
        )
    # Find an initial feasible scenario z and lambda
    z_initial, lambda_initial = find_initial_feasible_scenario_z_lambda(H, h, grid_size, is_silent)
    if is_milp
        lambda_initial = zeros((two_batt ? 3 : 2) * n_t)
    elseif !two_batt
        lambda_initial_full = lambda_initial
        lambda_initial = zeros(4 * n_t)
        for t in 1:n_t
            src = 6 * (t - 1)
            dst = 4 * (t - 1)
            lambda_initial[dst + 1:dst + 4] .= lambda_initial_full[src + 1:src + 4]
        end
    end

    # Initialize counters and logs
    gamma_star = 0.0
    n_star = 1.0
    m_star = two_batt ? 1.0 : 0.0
    s_star = 1.0
    u_star = zeros(n_u)
    sigma_star = 0.0
    w_star = copy(w_init)
    master_x_obj_history = fill(NaN, k1_max)
    u_star_history = fill(NaN, n_u, k1_max)
    w_scenarios_history = fill(NaN, n_w, k1_max)
    z_scenarios_history = fill(NaN, n_z, k2_max, k1_max)
    z_scenarios_history[:, 1, :] .= reshape(z_initial, :, 1, 1)
    n_lambda = is_milp ? (two_batt ? 3 : 2) * n_t : (two_batt ? 6 : 4) * n_t
    lambda_scenarios_history = fill(NaN, n_lambda, k2_max, k1_max)
    lambda_scenarios_history[:, 1, :] .= reshape(lambda_initial, :, 1, 1)
    master_x_solve_time_history = fill(NaN, k1_max)
    scenario_slot = 1
    scenarios_added = 1
    w_scenarios_history[:, scenario_slot] = w_init
    iter_log = NamedTuple[]
    n_master_solves = 0
    x_block_starts = Vector{Vector{Float64}}()
    solve_stats = Dict(
        :nlp_count => 0,
        :nlp_time => 0.0,
        :lp_count => 0,
        :lp_time => 0.0,
        :local_red_time => 0.0,
    )

    local_red_t0 = time()
    for iter in 1:k1_max
        println(repeat("=", 20), " SOLVE MASTER with ", scenario_slot, " scenarios ", repeat("=", 20))
        println("Iteration MASTER: ", iter, " / ", k1_max)
        # Solve the master problem
        print("Solving ", is_milp ? "MILP" : "NLP", "... ")
        flush(stdout)
        solve_t0 = time()
        optimize!(master_x)
        solve_elapsed = time() - solve_t0
        solve_stats[:nlp_count] += 1
        solve_stats[:nlp_time] += solve_elapsed
        println("Done.")

        # Check master solution status
        ts = termination_status(master_x)
        ps = primal_status(master_x)
        if ts != JuMP.MOI.OPTIMAL && ts != JuMP.MOI.LOCALLY_SOLVED && ts != JuMP.MOI.ALMOST_OPTIMAL && ts != JuMP.MOI.ALMOST_LOCALLY_SOLVED
            println("MASTER did not return OPTIMAL.")
            println("Termination status: ", ts)
            println("Primal status: ", ps)
            println("Stopping local reduction.")
            break
        end

        # Get the optimal master solution and objective value
        gamma_star = value(master_x[:gamma])
        n_star = value(master_x[:n])
        m_star = two_batt ? value(master_x[:m]) : 0.0
        s_star = value(master_x[:s])
        n_master_solves += 1
        master_x_obj_history[n_master_solves] = gamma_star
        master_x_solve_time_history[n_master_solves] = solve_elapsed
        u_star = vec(value.(master_x[:u]))
        x_block_starts = [value.(x_blocks[k]) for k in 1:scenario_slot]
        z_block_starts = [value.(z_blocks[k]) for k in 1:scenario_slot]
        l_block_starts = [value.(l_blocks[k]) for k in 1:scenario_slot]
        u_star_history[:, n_master_solves] = u_star

        println("Optimal value: ", gamma_star)
        println("Master solve time: ", round(solve_elapsed; digits=3), " s")
        println()

        # Check if we have reached the maximum number of scenario slots for the master problem
        if scenario_slot + 1 > k1_slots
            println("Reached maximum number of scenario slots. Stopping.")
            println()
            break
        end
        scenarios_added_this_iter = 0

        if use_grid_selector_oracle
            grid_sigma_star, w_grid_star, grid_solve_elapsed, grid_ts, grid_ps, _ = solve_grid_feasibility_selector_oracle(
                P,
                p,
                grid_size,
                u_star,
                s_star,
                pg_max,
                milp_optimizer=milp_optimizer,
                is_silent=is_silent,
                two_batt=two_batt,
            )
            grid_sigmas = [grid_sigma_star]
            w_grid_candidates = reshape(w_grid_star, :, 1)
            grid_n_solves = 1
        else
            grid_sigmas, w_grid_candidates, grid_solve_elapsed, grid_ts, grid_ps, grid_n_solves = solve_grid_feasibility_oracle!(
                grid_oracle_models,
                u_star,
                s_star,
                pg_max,
                two_batt,
            )
        end
        solve_stats[:lp_count] += grid_n_solves
        solve_stats[:lp_time] += grid_solve_elapsed
        if use_grid_selector_oracle
            println("Grid feasibility selector oracle | sigma* = ", grid_sigmas[1])
        else
            println("Grid feasibility oracle | sigma+* = ", grid_sigmas[1], " | sigma-* = ", grid_sigmas[2])
        end
        println("Grid feasibility oracle solves = ", grid_n_solves)
        println("Grid feasibility oracle solve time: ", round(grid_solve_elapsed; digits=3), " s")
        if !(grid_ts in (JuMP.MOI.OPTIMAL, JuMP.MOI.ALMOST_OPTIMAL))
            println("Grid feasibility oracle status: ", grid_ts, " | primal status: ", grid_ps)
        end
        println()

        stop_after_grid_oracle = false
        for grid_side in eachindex(grid_sigmas)
            if grid_sigmas[grid_side] <= tol_1
                continue
            end
            if scenario_slot + 1 > k1_slots
                println("Reached maximum number of scenario slots before adding grid-feasibility scenario. Stopping.")
                println()
                stop_after_grid_oracle = true
                break
            end
            w_grid_star = w_grid_candidates[:, grid_side]
            scenario_slot += 1
            add_master_scenario!(
                master_x,
                x_blocks,
                z_blocks,
                l_blocks,
                scenario_slot,
                w_grid_star,
                nothing,
                x_block_starts,
                z_block_starts,
                l_block_starts,
                u,
                gamma,
                n,
                s,
                u_star,
                gamma_star,
                n_star,
                m_star,
                s_star,
                grid_size,
                x_dim_factor,
                u_dim,
                is_milp,
                two_batt,
                is_cycle,
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
                battery_cost,
                pv_cost,
                base_pv_capacity_kw,
                delta_t,
                e_batt,
                energy_scale,
                cost_scale,
            )
            push!(x_block_starts, [something(start_value(v), 0.0) for v in x_blocks[scenario_slot]])
            push!(z_block_starts, [something(start_value(v), 0.0) for v in z_blocks[scenario_slot]])
            push!(l_block_starts, [something(start_value(v), 0.0) for v in l_blocks[scenario_slot]])
            scenarios_added += 1
            scenarios_added_this_iter += 1
            w_scenarios_history[:, scenario_slot] = w_grid_star
            grid_label = use_grid_selector_oracle ? "GRID_SELECTOR" : (grid_side == 1 ? "GRID+" : "GRID-")
            println("Added scenario w", scenario_slot, " - ", grid_label)
            println()

            if scenario_slot + 1 > k1_slots
                println("Reached maximum number of scenario slots after grid-feasibility scenario. Stopping.")
                println()
                stop_after_grid_oracle = true
                break
            end
        end
        stop_after_grid_oracle && break

        # Run the local reduction subroutine to find a violating scenario w (maximization step)
        sigma_star, w_star, x_witness, z_scenarios_used,  z_scenarios_history, lambda_scenarios_history = solve_local_red_z_lambda(
                max_z_model,
                z_initial, 
                lambda_initial,
                z_scenarios_history,
                lambda_scenarios_history,
                iter,
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
                tol_2,
                is_silent=is_silent,
                z_master_restarts=z_master_restarts,
                z_master_restart_seed=z_master_restart_seed,
                show_z_iter=show_z_iter,
                show_w_vec=show_w_vec,
                solve_stats=solve_stats,
                is_milp=is_milp,
                milp_optimizer=milp_optimizer,
                is_cycle=is_cycle,
        )

        # Check constraint violation
        if sigma_star <= tol_1
            push!(iter_log, (
                iteration=iter,
                gamma=gamma_star,
                sigma=sigma_star,
                u_star=u_star,
                master_x_solve_time=solve_elapsed,
                scenarios_added_this_iter=scenarios_added_this_iter,
                cumulative_scenarios=scenarios_added,
                z_scenarios_used=z_scenarios_used,
            ))
            if scenarios_added_this_iter > 0
                println("No constraint violation at MASTER level (sigma <= $tol_1); re-solving after added scenario.")
                println()
                continue
            end
            println("No constraint violation at MASTER level (sigma <= $tol_1).")
            println()
            break
        end

        println("Z summary | sigma* = ", sigma_star)
        println("Z summary | z scenarios used = ", z_scenarios_used)
        if show_w_vec
            println("Z summary | w* = ", w_star)
        end

        # Add the new scenario w to the master problem
        scenario_slot += 1
        x_blocks[scenario_slot] = @variable(master_x, [1:x_dim_factor * grid_size])
        z_blocks[scenario_slot] = @variable(master_x, [1:2 * (grid_size - 1)], lower_bound = 0)
        l_dim_added = is_milp ? (two_batt ? 3 : 2) : (two_batt ? 6 : 4)
        if is_milp
            l_blocks[scenario_slot] = @variable(master_x, [1:l_dim_added * (grid_size - 1)], Bin)
        else
            l_blocks[scenario_slot] = @variable(master_x, [1:l_dim_added * (grid_size - 1)], lower_bound=0)
        end
        @constraint(master_x, H * vec(z_blocks[scenario_slot]) .<= h)
        n_var = master_x[:n]

        if is_milp
            @constraint(master_x, [t in 1:grid_size], x_blocks[scenario_slot][t] >= soc_min * n_var * e_batt)
            @constraint(master_x, [t in 1:grid_size], x_blocks[scenario_slot][t] <= soc_max * n_var * e_batt)
        else
            @constraint(master_x, [t in 1:grid_size], (x_blocks[scenario_slot][t] - soc_min * n_var * e_batt) / energy_scale >= 0)
            @constraint(master_x, [t in 1:grid_size], (x_blocks[scenario_slot][t] - soc_max * n_var * e_batt) / energy_scale <= 0)
        end
        if two_batt
            m_var = master_x[:m]
            if is_milp
                @constraint(master_x, [t in 1:grid_size], x_blocks[scenario_slot][grid_size + t] >= soc_min * m_var * e_batt)
                @constraint(master_x, [t in 1:grid_size], x_blocks[scenario_slot][grid_size + t] <= soc_max * m_var * e_batt)
            else
                @constraint(master_x, [t in 1:grid_size], (x_blocks[scenario_slot][grid_size + t] - soc_min * m_var * e_batt) / energy_scale >= 0)
                @constraint(master_x, [t in 1:grid_size], (x_blocks[scenario_slot][grid_size + t] - soc_max * m_var * e_batt) / energy_scale <= 0)
            end
        end

        # Initial conditions
        if is_milp
            # @constraint(master_x, x_blocks[scenario_slot][1] == soc0 * n_var * e_batt)
            @constraint(master_x, x_blocks[scenario_slot][1] == soc0_var * n_var * e_batt)
        else
            @constraint(master_x, (x_blocks[scenario_slot][1] - soc0 * n_var * e_batt) / energy_scale == 0)
        end
        if is_cycle
            if is_milp
                @constraint(master_x, x_blocks[scenario_slot][grid_size] == soc0 * n_var * e_batt)
            else
                @constraint(master_x, (x_blocks[scenario_slot][grid_size] - soc0 * n_var * e_batt) / energy_scale == 0)
            end
        end
        if two_batt
            if is_milp
                @constraint(master_x, x_blocks[scenario_slot][grid_size + 1] == soc0 * m_var * e_batt)
            else
                @constraint(master_x, (x_blocks[scenario_slot][grid_size + 1] - soc0 * m_var * e_batt) / energy_scale == 0)
            end
            if is_cycle
                if is_milp
                    @constraint(master_x, x_blocks[scenario_slot][2 * grid_size] == soc0 * m_var * e_batt)
                else
                    @constraint(master_x, (x_blocks[scenario_slot][2 * grid_size] - soc0 * m_var * e_batt) / energy_scale == 0)
                end
            end
        end

        # Constraints
        for t in 1:grid_size-1
            z_idx = 2 * (t - 1)
            if is_milp
                l_idx = (two_batt ? 3 : 2) * (t - 1)
                @constraint(master_x, u[1, t] <= n_max * pc_max * l_blocks[scenario_slot][l_idx + 1])
                @constraint(master_x, u[2, t] <= n_max * pd_max * (1 - l_blocks[scenario_slot][l_idx + 1]))
                if two_batt
                    @constraint(master_x, u[3, t] <= m_max * pc_max * l_blocks[scenario_slot][l_idx + 2])
                    @constraint(master_x, u[4, t] <= m_max * pd_max * (1 - l_blocks[scenario_slot][l_idx + 2]))
                    @constraint(master_x, z_blocks[scenario_slot][z_idx + 1] <= pg_max * l_blocks[scenario_slot][l_idx + 3])
                    @constraint(master_x, z_blocks[scenario_slot][z_idx + 2] <= pg_max * (1 - l_blocks[scenario_slot][l_idx + 3]))
                else
                    @constraint(master_x, z_blocks[scenario_slot][z_idx + 1] <= pg_max * l_blocks[scenario_slot][l_idx + 2])
                    @constraint(master_x, z_blocks[scenario_slot][z_idx + 2] <= pg_max * (1 - l_blocks[scenario_slot][l_idx + 2]))
                end
            else
                l_idx = (two_batt ? 6 : 4) * (t - 1)

                # simplex constraints on lambda
                @constraint(master_x, l_blocks[scenario_slot][l_idx + 1] + l_blocks[scenario_slot][l_idx + 2] == 1)
                @constraint(master_x, l_blocks[scenario_slot][l_idx + 3] + l_blocks[scenario_slot][l_idx + 4] == 1)
                if two_batt
                    @constraint(master_x, l_blocks[scenario_slot][l_idx + 5] + l_blocks[scenario_slot][l_idx + 6] == 1)
                end

                # logic constraints
                @constraint(master_x, l_blocks[scenario_slot][l_idx + 1] * u[1, t] + l_blocks[scenario_slot][l_idx + 2] * u[2, t] <= 0)
                @constraint(master_x, l_blocks[scenario_slot][l_idx + 3] * z_blocks[scenario_slot][z_idx + 1] + l_blocks[scenario_slot][l_idx + 4] * z_blocks[scenario_slot][z_idx + 2] <= 0)
                if two_batt
                    @constraint(master_x, l_blocks[scenario_slot][l_idx + 5] * u[3, t] + l_blocks[scenario_slot][l_idx + 6] * u[4, t] <= 0)
                end
            end

            # power balance
            idx = 6 * (t - 1)
            PL_t = w_star[idx + 1]
            PR_t = w_star[idx + 2]
            eta_ch_t = w_star[idx + 5]
            rho_dis_t = w_star[idx + 6]
            batt2_net_t = two_batt ? u[4, t] - u[3, t] : 0.0
            @constraint(
                master_x,
                z_blocks[scenario_slot][z_idx + 1] - z_blocks[scenario_slot][z_idx + 2] + u[2, t] - u[1, t] + batt2_net_t - PL_t + s * PR_t == 0
            )

            # State evolution
            if is_milp
                @constraint(
                    master_x,
                    x_blocks[scenario_slot][t + 1] == x_blocks[scenario_slot][t] + (eta_ch_t * u[1, t] - rho_dis_t * u[2, t]) * delta_t
                )
            else
                @constraint(
                    master_x,
                    (x_blocks[scenario_slot][t + 1] - x_blocks[scenario_slot][t] - (eta_ch_t * u[1, t] - rho_dis_t * u[2, t]) * delta_t) / energy_scale == 0
                )
            end
            if two_batt
                if is_milp
                    @constraint(
                        master_x,
                        x_blocks[scenario_slot][grid_size + t + 1] == x_blocks[scenario_slot][grid_size + t] + (eta_ch_t * u[3, t] - rho_dis_t * u[4, t]) * delta_t
                    )
                else
                    @constraint(
                        master_x,
                        (x_blocks[scenario_slot][grid_size + t + 1] - x_blocks[scenario_slot][grid_size + t] - (eta_ch_t * u[3, t] - rho_dis_t * u[4, t]) * delta_t) / energy_scale == 0
                    )
                end
            end
        end
        battery_count_var = n
        if two_batt
            battery_count_var += master_x[:m]
        end
        @constraint(
            master_x,
            (battery_count_var * e_batt * battery_cost + s * base_pv_capacity_kw * pv_cost +
            sum(
                delta_t * (
                    w_star[6 * (t - 1) + 3] * z_blocks[scenario_slot][2 * (t - 1) + 1] -
                    w_star[6 * (t - 1) + 4] * z_blocks[scenario_slot][2 * (t - 1) + 2]
                )
                for t in 1:grid_size-1
            ) - gamma) / cost_scale <= 0
        )

        set_start_value(master_x[:gamma], gamma_star)
        set_start_value(master_x[:n], n_star)
        if two_batt
            set_start_value(master_x[:m], m_star)
        end
        set_start_value(master_x[:s], s_star)
        for t in 1:(grid_size - 1)
            u_idx = u_dim * (t - 1)
            for j in 1:u_dim
                set_start_value(u[j, t], u_star[u_idx + j])
            end
        end
        for k in 1:(scenario_slot - 1)
            for i in eachindex(x_blocks[k])
                set_start_value(x_blocks[k][i], x_block_starts[k][i])
            end
            for i in eachindex(z_blocks[k])
                set_start_value(z_blocks[k][i], z_block_starts[k][i])
            end
            for i in eachindex(l_blocks[k])
                set_start_value(l_blocks[k][i], l_block_starts[k][i])
            end
        end

        initialize_new_master_scenario!(
            master_x,
            x_blocks[scenario_slot],
            z_blocks[scenario_slot],
            l_blocks[scenario_slot],
            u_star,
            w_star,
            gamma_star,
            n_star,
            m_star,
            s_star,
            soc0,
            delta_t,
            e_batt,
            x_witness=x_witness,
            two_batt=two_batt,
            is_milp=is_milp,
        )

        scenarios_added += 1
        scenarios_added_this_iter += 1
        w_scenarios_history[:, scenario_slot] = w_star
        println("Added scenario w", scenario_slot, " - MASTER")

        push!(iter_log, (
            iteration=iter,
            gamma=gamma_star,
            sigma=sigma_star,
            u_star=u_star,
            master_x_solve_time=solve_elapsed,
            scenarios_added_this_iter=scenarios_added_this_iter,
            cumulative_scenarios=scenarios_added,
            z_scenarios_used=z_scenarios_used,
        ))
        println()
    end

    local_red_elapsed = time() - local_red_t0
    solve_stats[:local_red_time] += local_red_elapsed

    # Print final summary of results
    master_x_obj_history_used = master_x_obj_history[1:n_master_solves]
    u_star_history_used = u_star_history[:, 1:n_master_solves]
    w_scenarios_history_used = w_scenarios_history[:, 1:scenario_slot]
    z_scenarios_history_used = z_scenarios_history[:, :, 1:n_master_solves]
    lambda_scenarios_history_used = lambda_scenarios_history[:, :, 1:n_master_solves]
    master_x_solve_time_history_used = master_x_solve_time_history[1:n_master_solves]
    total_elapsed_seconds = time() - total_t0
    print_final_summary(gamma_star, scenarios_added, solve_stats)

    results_path = nothing
    if save_results
        resolved_results_dir = String(results_dir)
        resolved_data_path = problem_source isa AbstractString ? problem_source : nothing
        source_label = "read_microgrid_problem_data"
        data_config = (
            processed_data_path=processed_data_path,
            horizon_days=horizon_days,
            initial_day=initial_day,
            soc0=soc0,
            load_error_pct=load_error_pct,
            pv_error_pct=pv_error_pct,
            pv_factor=pv_factor,
            grid_power_factor=grid_power_factor,
            battery_factor=battery_factor,
            two_batt=two_batt,
            is_cycle=is_cycle,
            use_grid_selector_oracle=use_grid_selector_oracle,
            n_max=n_max,
            m_max=m_max,
            s_max=s_max,
        )
        results_path = _save_local_reduction_results(
            resolved_results_dir,
            resolved_data_path,
            source_label,
            soc0,
            horizon_days,
            grid_power_factor,
            is_milp,
            two_batt,
            is_cycle,
            master_x_obj_history_used,
            u_star_history_used,
            w_scenarios_history_used,
            z_scenarios_history_used,
            lambda_scenarios_history_used,
            scenarios_added,
            solve_stats,
            iter_log,
            master_x_solve_time_history_used,
            total_elapsed_seconds,
            n_star,
            m_star,
            s_star,
            uncertainty_bounds,
            microgrid_params,
            data_config,
        )
        println("Saved local reduction results to: ", results_path)
    end

    return (
        model=master_x,
        u_star=u_star,
        n_star=n_star,
        m_star=m_star,
        s_star=s_star,
        gamma_star=gamma_star,
        uncertainty_bounds=uncertainty_bounds,
        microgrid_params=microgrid_params,
        data_config=(
            processed_data_path=processed_data_path,
            horizon_days=horizon_days,
            initial_day=initial_day,
            soc0=soc0,
            load_error_pct=load_error_pct,
            pv_error_pct=pv_error_pct,
            pv_factor=pv_factor,
            grid_power_factor=grid_power_factor,
            battery_factor=battery_factor,
            two_batt=two_batt,
            is_cycle=is_cycle,
            use_grid_selector_oracle=use_grid_selector_oracle,
            n_max=n_max,
            m_max=m_max,
            s_max=s_max,
        ),
        scenarios_added=scenarios_added,
        solve_stats=solve_stats,
        iteration_log=iter_log,
        master_objective_history=master_x_obj_history_used,
        u_star_history=u_star_history_used,
        x_star_scenarios=x_block_starts,
        w_scenarios_history=w_scenarios_history_used,
        z_scenarios_history=z_scenarios_history_used,
        lambda_scenarios_history=lambda_scenarios_history_used,
        master_x_solve_time_history=master_x_solve_time_history_used,
        results_path=results_path,
        total_elapsed_seconds=total_elapsed_seconds,
    )
end
