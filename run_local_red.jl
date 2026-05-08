using JuMP
using LinearAlgebra
using Dates
using JLD2
using Random

include("load_problem_data.jl")
include("print_initial_summary.jl")
include("print_final_summary.jl")
include("build_master_problem.jl")
include("find_initial_feasible_scenario_master.jl")
include("find_initial_feasible_scenario_z_lambda.jl")
include("build_max_z.jl")
include("solve_local_red_z_lambda.jl")
include("initialize_new_master_scenario.jl")
include("add_master_scenario.jl")
include("build_grid_feasibility_selector_oracle.jl")

function _save_local_reduction_results(
    results_dir::AbstractString,
    resolved_data_path,
    source_label,
    soc0,
    horizon_days,
    grid_power_factor,
    is_mip,
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
    s_star,
    uncertainty_bounds,
    microgrid_params,
    data_config,
)
    mkpath(results_dir)
    timestamp = Dates.format(now(), "yyyy-mm-dd_HHMMSS")
    config_stem = _run_config_file_stem(soc0, horizon_days, grid_power_factor)
    if is_mip
        formulation_stem = "MIP"
    else
        formulation_stem = "NLP"
    end
    stem = formulation_stem * "_" * config_stem
    results_path = joinpath(results_dir, stem * "_" * timestamp * ".jld2")
    JLD2.jldsave(
        results_path;
        resolved_data_path=resolved_data_path === nothing ? "" : String(resolved_data_path),
        data_source=String(source_label),
        saved_at=string(now()),
        is_mip=is_mip,
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
        s_star=s_star,
        uncertainty_bounds=uncertainty_bounds,
        microgrid_params=microgrid_params,
        data_config=data_config,
    )
    return results_path
end

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
)
    result_path = warm_start_file === nothing ?
        _find_latest_1day_result(results_dir, soc0, grid_power_factor) :
        (isabspath(String(warm_start_file)) ? String(warm_start_file) : joinpath(results_dir, String(warm_start_file)))

    result_path === nothing && return nothing
    isfile(result_path) || return nothing

    data = JLD2.load(result_path)

    u_hist = data["u_star_history"]
    col = _last_finite_column(u_hist)
    col === nothing && return nothing

    u_day = u_hist[:, col]
    gamma_start = haskey(data, "master_x_obj_history") ? _last_finite_value(data["master_x_obj_history"]) : nothing
    n_start = haskey(data, "n_star") ? data["n_star"] : 1.0
    s_start = haskey(data, "s_star") ? data["s_star"] : 1.0

    return (
        result_path=result_path,
        u_start=_repeat_control_to_horizon(u_day, n_t_target, u_dim),
        gamma_start=gamma_start,
        n_start=n_start,
        s_start=s_start,
    )
end

function _local_red_solution_ok(model)
    ts = termination_status(model)
    ps = primal_status(model)
    return (ts in (
            JuMP.MOI.OPTIMAL,
            JuMP.MOI.LOCALLY_SOLVED,
            JuMP.MOI.ALMOST_OPTIMAL,
            JuMP.MOI.ALMOST_LOCALLY_SOLVED,
        )) && (ps in (
            JuMP.MOI.FEASIBLE_POINT,
            JuMP.MOI.NEARLY_FEASIBLE_POINT,
        ))
end

function _randomize_master_start!(
    master_x,
    x_blocks,
    z_blocks,
    l_blocks,
    scenario_slot,
    w_scenarios_history,
    rng,
    grid_size,
    soc0,
    pc_max,
    pd_max,
    pg_max,
    n_max,
    s_max,
    delta_t,
    e_batt,
    cost_scale,
    is_mip,
)
    n_t = grid_size - 1
    u = master_x[:u]
    n_start = is_mip ? rand(rng, 1:floor(Int, n_max)) : 1.0 + rand(rng) * (n_max - 1.0)
    s_start = is_mip ? rand(rng, 1:floor(Int, s_max)) : 1.0 + rand(rng) * (s_max - 1.0)
    set_start_value(master_x[:n], n_start)
    set_start_value(master_x[:s], s_start)
    set_start_value(master_x[:gamma], cost_scale * (1.0 + 10.0 * rand(rng)))

    for t in 1:n_t
        set_start_value(u[1, t], rand(rng) * n_start * pc_max)
        set_start_value(u[2, t], rand(rng) * n_start * pd_max)
    end

    for k in 1:scenario_slot
        w_k = w_scenarios_history[:, k]
        if !all(isfinite, w_k)
            continue
        end

        set_start_value(x_blocks[k][1], soc0 * n_start * e_batt)
        for t in 1:n_t
            u_idx = 2 * (t - 1)
            w_idx = 6 * (t - 1)
            z_idx = 2 * (t - 1)
            pc_t = something(start_value(u[1, t]), 0.0)
            pd_t = something(start_value(u[2, t]), 0.0)
            set_start_value(z_blocks[k][z_idx + 1], rand(rng) * pg_max)
            set_start_value(z_blocks[k][z_idx + 2], rand(rng) * pg_max)
            set_start_value(
                x_blocks[k][t + 1],
                something(start_value(x_blocks[k][t]), soc0 * n_start * e_batt) +
                (w_k[w_idx + 5] * pc_t - w_k[w_idx + 6] * pd_t) * delta_t,
            )

            if is_mip
                l_idx = 2 * (t - 1)
                set_start_value(l_blocks[k][l_idx + 1], rand(rng, 0:1))
                set_start_value(l_blocks[k][l_idx + 2], rand(rng, 0:1))
            else
                l_idx = 4 * (t - 1)
                a = rand(rng)
                b = rand(rng)
                set_start_value(l_blocks[k][l_idx + 1], a)
                set_start_value(l_blocks[k][l_idx + 2], 1.0 - a)
                set_start_value(l_blocks[k][l_idx + 3], b)
                set_start_value(l_blocks[k][l_idx + 4], 1.0 - b)
            end
        end
    end

    return nothing
end

function _solve_master_with_recovery!(
    master_x,
    x_blocks,
    z_blocks,
    l_blocks,
    scenario_slot,
    w_scenarios_history,
    solve_stats,
    rng,
    grid_size,
    soc0,
    pc_max,
    pd_max,
    pg_max,
    n_max,
    s_max,
    delta_t,
    e_batt,
    cost_scale,
    is_mip,
    max_recovery_restarts,
)
    total_elapsed = 0.0
    last_error = nothing
    n_attempts = max_recovery_restarts + 1

    for attempt in 1:n_attempts
        if attempt > 1
            println("Master solve failed to return a feasible point; random restart ", attempt - 1, " / ", max_recovery_restarts, ".")
            _randomize_master_start!(
                master_x,
                x_blocks,
                z_blocks,
                l_blocks,
                scenario_slot,
                w_scenarios_history,
                rng,
                grid_size,
                soc0,
                pc_max,
                pd_max,
                pg_max,
                n_max,
                s_max,
                delta_t,
                e_batt,
                cost_scale,
                is_mip,
            )
        end

        solve_t0 = time()
        try
            optimize!(master_x)
            last_error = nothing
        catch err
            last_error = err
        end
        solve_elapsed = time() - solve_t0
        total_elapsed += solve_elapsed
        if is_mip
            solve_stats[:mip_count] += 1
            solve_stats[:mip_time] += solve_elapsed
        else
            solve_stats[:nlp_count] += 1
            solve_stats[:nlp_time] += solve_elapsed
        end

        if last_error === nothing && _local_red_solution_ok(master_x)
            return true, total_elapsed, nothing
        end

        if last_error !== nothing
            println("Master solve threw an error: ", sprint(showerror, last_error))
        else
            println("Master status: ", termination_status(master_x), " | primal: ", primal_status(master_x))
        end
    end

    return false, total_elapsed, last_error
end

function run_local_red(
    problem_source=nothing;
    processed_data_path::AbstractString=joinpath(@__DIR__, "Processed_Data_15min.csv"),
    horizon_days::Int=1,
    initial_day="2016-05-01",
    soc0::Float64=0.4,
    load_error_pct::Float64=0.10,
    pv_error_pct::Float64=0.10,
    pv_factor::Float64=1.0,
    grid_power_factor::Float64=0.2,
    battery_cost::Float64=200.0,
    pv_cost::Float64=1500.0,
    base_pv_capacity_kw::Float64=90.84,
    k1_max::Int=5,
    k2_max::Int=15,
    ipopt_lin_solver="ma57",
    mip_optimizer="Gurobi",
    is_mip::Bool=false,
    max_iter_ipopt=5000,
    tol_1::Float64=1e-6,
    tol_2::Float64=1e-6,
    z_master_restarts::Int=5,
    z_master_restart_seed::Int=1234,
    master_recovery_restarts::Int=20,
    z_master_recovery_restarts::Int=20,
    show_w_vec::Bool=false,
    show_z_iter::Bool=true,
    is_silent::Bool=false,
    save_results::Bool=false,
    results_dir::AbstractString=joinpath(@__DIR__, "Results"),
    warm_start_from_1day::Bool=false,
    warm_start_1day_file=nothing,
    is_cycle::Bool=false,
    n_max::Real=100,
    s_max::Real=100,
)
    """
    Local reduction algorithm for the min-max-min problem.
    """
    if min(k1_max, k2_max) < 2
        error("k1_max and k2_max must all be at least 2.")
    end
    if z_master_restarts < 1
        error("z_master_restarts must be at least 1.")
    end
    if min(master_recovery_restarts, z_master_recovery_restarts) < 0
        error("Recovery restart counts must be nonnegative.")
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
    u_dim = 2
    n_u = u_dim * n_t
    n_x = grid_size
    m_max = 0.0
    soc0 = microgrid_params["SoC0"]
    soc_min = microgrid_params["SOC_MIN"]
    soc_max = microgrid_params["SOC_MAX"]
    pc_max = microgrid_params["Pc_MAX"]
    pd_max = microgrid_params["Pd_MAX"]
    pg_max = microgrid_params["PG_plus_MAX"]
    delta_t = microgrid_params["Δt"]
    e_batt = microgrid_params["E_BATT"]
    cost_scale = max(abs(base_pv_capacity_kw * pv_cost), abs(e_batt * battery_cost), 1.0)
    energy_scale = max(abs(e_batt * n_max), 1.0)
    println("Done.")

    k1_slots = k1_max

    print_initial_summary(
        n_x,
        n_u,
        n_z,
        n_w,
        k1_max,
        k2_max,
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
    master_x, x_blocks, z_blocks, l_blocks = build_master_problem(
            is_mip,
            mip_optimizer,
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
            pg_max,
            battery_cost,
            pv_cost,
            base_pv_capacity_kw,
            delta_t,
            e_batt,
            is_silent;
            w_init=w_init,
            is_cycle=is_cycle,
            n_max=n_max,
            s_max=s_max,
        )

    u = master_x[:u]
    gamma = master_x[:gamma]
    n = master_x[:n]
    s = master_x[:s]

    if warm_start_from_1day && horizon_days > 1
        warm_start = _load_1day_warm_start(
            results_dir,
            soc0,
            grid_power_factor,
            n_t;
            warm_start_file=warm_start_1day_file,
            u_dim=u_dim,
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
                warm_start.s_start,
                soc0,
                delta_t,
                e_batt,
                is_mip=is_mip,
            )
            println("Warm-started master from 1-day result: ", warm_start.result_path)
        end
    end

    # Build the z maximization problem
    max_z_model = build_max_z(
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
    
    
    # max_grid_model = build_grid_feasibility_selector_oracle(
    #                         is_mip,
    #                         mip_optimizer,
    #                         ipopt_lin_solver, 
    #                         max_iter_ipopt,
    #                         P,
    #                         p,
    #                         grid_size,
    #                         pg_max,
    #                         pc_max=pc_max,
    #                         pd_max=pd_max,
    #                         n_max=n_max,
    #                         s_max=s_max,
    #                         is_silent=is_silent,
    #                     )

    

    # Find an initial feasible scenario z and lambda
    z_initial, lambda_initial = find_initial_feasible_scenario_z_lambda(H, h, grid_size, is_silent)
    if is_mip
        lambda_initial = zeros(2 * n_t)
    end

    # Initialize counters and logs
    gamma_star = 0.0
    n_star = 1.0
    s_star = 1.0
    u_star = zeros(n_u)
    sigma_star = 0.0
    w_star = copy(w_init)
    master_x_obj_history = fill(NaN, k1_max)
    u_star_history = fill(NaN, n_u, k1_max)
    w_scenarios_history = fill(NaN, n_w, k1_max)
    z_scenarios_history = fill(NaN, n_z, k2_max, k1_max)
    z_scenarios_history[:, 1, :] .= reshape(z_initial, :, 1, 1)
    n_lambda = is_mip ? 2 * n_t : 4 * n_t
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
        :mip_count => 0,
        :mip_time => 0.0,
        :local_red_time => 0.0,
    )

    local_red_t0 = time()
    master_restart_rng = MersenneTwister(z_master_restart_seed + 10_000)
    for iter in 1:k1_max
        println(repeat("=", 20), " SOLVE MASTER with ", scenario_slot, " scenarios ", repeat("=", 20))
        println("Iteration MASTER: ", iter, " / ", k1_max)
        # Solve the master problem
        print("Solving ", is_mip ? "MIP" : "NLP", "... ")
        flush(stdout)
        master_ok, solve_elapsed, master_error = _solve_master_with_recovery!(
            master_x,
            x_blocks,
            z_blocks,
            l_blocks,
            scenario_slot,
            w_scenarios_history,
            solve_stats,
            master_restart_rng,
            grid_size,
            soc0,
            pc_max,
            pd_max,
            pg_max,
            n_max,
            s_max,
            delta_t,
            e_batt,
            cost_scale,
            is_mip,
            master_recovery_restarts,
        )
        println("Done.")

        # Check master solution status
        ts = termination_status(master_x)
        ps = primal_status(master_x)
        if !master_ok
            println("MASTER did not return a feasible solution after ", master_recovery_restarts, " recovery restarts.")
            if master_error !== nothing
                println("Last error: ", sprint(showerror, master_error))
            else
                println("Termination status: ", ts)
                println("Primal status: ", ps)
            end
            println("Stopping local reduction.")
            break
        end

        # Get the optimal master solution and objective value
        gamma_star = value(master_x[:gamma])
        n_star = value(master_x[:n])
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

        # # Maximize grid constraints
        # print("Maximizing grid constraints... ")
        # flush(stdout)
        # fix.(max_grid_model[:u_star_var], u_star, force = true)
        # fix(max_grid_model[:s_star_var], s_star, force = true)
        # solve_t0 = time()
        # optimize!(max_grid_model)
        # solve_elapsed = time() - solve_t0
        # if is_mip
        #     solve_stats[:mip_count] += 1
        #     solve_stats[:mip_time] += solve_elapsed 
        # else
        #     solve_stats[:nlp_count] += 1
        #     solve_stats[:nlp_time] += solve_elapsed
        # end
        # println("Done.")

        # # Check master solution status
        # ts = termination_status(max_grid_model)
        # ps = primal_status(max_grid_model)
        # if ts != JuMP.MOI.OPTIMAL && ts != JuMP.MOI.LOCALLY_SOLVED && ts != JuMP.MOI.ALMOST_OPTIMAL && ts != JuMP.MOI.ALMOST_LOCALLY_SOLVED
        #     println("MASTER did not return OPTIMAL.")
        #     println("Termination status: ", ts)
        #     println("Primal status: ", ps)
        #     println("Stopping local reduction.")
        #     break
        # end

        # w_grid_star = vec(value.(max_grid_model[:w]))
        # grid_sigma_star = value(max_grid_model[:sigma])
        # println("Grid feasibility selector oracle | sigma* = ", grid_sigma_star)
        # println()

        # if grid_sigma_star <= tol_1
        #     println("No violated grid constraint found.")
        #     println()
        # elseif scenario_slot + 1 > k1_slots
        #     println("Reached maximum number of scenario slots before adding grid-feasibility scenario. Stopping.")
        #     println()
        #     break
        # else
        #     scenario_slot += 1
        #     starts = add_master_scenario!(
        #         master_x,
        #         x_blocks,
        #         z_blocks,
        #         l_blocks,
        #         scenario_slot,
        #         w_grid_star,
        #         nothing,
        #         x_block_starts,
        #         z_block_starts,
        #         l_block_starts,
        #         u,
        #         gamma,
        #         n,
        #         s,
        #         u_star,
        #         gamma_star,
        #         n_star,
        #         s_star,
        #         grid_size,
        #         u_dim,
        #         is_mip,
        #         is_cycle,
        #         H,
        #         h,
        #         soc0,
        #         soc_min,
        #         soc_max,
        #         pc_max,
        #         pd_max,
        #         pg_max,
        #         n_max,
        #         battery_cost,
        #         pv_cost,
        #         base_pv_capacity_kw,
        #         delta_t,
        #         e_batt,
        #         energy_scale,
        #         cost_scale,
        #     )
        #     push!(x_block_starts, starts.x_start)
        #     push!(z_block_starts, starts.z_start)
        #     push!(l_block_starts, starts.l_start)
        #     scenarios_added += 1
        #     scenarios_added_this_iter += 1
        #     w_scenarios_history[:, scenario_slot] = w_grid_star
        #     println("Added scenario w", scenario_slot, " - GRID")
        #     println()
        # end

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
                tol_2,
                is_silent=is_silent,
                z_master_restarts=z_master_restarts,
                z_master_restart_seed=z_master_restart_seed,
                z_master_recovery_restarts=z_master_recovery_restarts,
                show_z_iter=show_z_iter,
                show_w_vec=show_w_vec,
                solve_stats=solve_stats,
                is_mip=is_mip,
                mip_optimizer=mip_optimizer,
                is_cycle=is_cycle,
        )

        println("Z summary | sigma* = ", sigma_star)
        println("Z summary | z scenarios used = ", z_scenarios_used)
        if show_w_vec
            println("Z summary | w* = ", w_star)
        end

        if sigma_star < tol_1
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
                println("No constraint violation at MASTER level (sigma < $tol_1); re-solving after added grid scenario.")
                println()
                continue
            end
            println("No constraint violation at MASTER level (sigma < $tol_1).")
            println()
            break
        elseif scenario_slot + 1 > k1_slots
            println("Reached maximum number of scenario slots before adding master scenario. Stopping.")
            println()
            break
        else
            scenario_slot += 1
            starts = add_master_scenario!(
                master_x,
                x_blocks,
                z_blocks,
                l_blocks,
                scenario_slot,
                w_star,
                x_witness,
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
                s_star,
                grid_size,
                u_dim,
                is_mip,
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
                battery_cost,
                pv_cost,
                base_pv_capacity_kw,
                delta_t,
                e_batt,
                energy_scale,
                cost_scale,
            )
            push!(x_block_starts, starts.x_start)
            push!(z_block_starts, starts.z_start)
            push!(l_block_starts, starts.l_start)

            scenarios_added += 1
            scenarios_added_this_iter += 1
            w_scenarios_history[:, scenario_slot] = w_star
            println("Added scenario w", scenario_slot, " - MASTER")
        end

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
            is_cycle=is_cycle,
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
            is_mip,
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
            is_cycle=is_cycle,
            n_max=n_max,
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
