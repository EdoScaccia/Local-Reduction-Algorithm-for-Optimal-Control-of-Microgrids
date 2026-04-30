using JuMP

include("load_problem_data.jl")
include("build_master_problem.jl")
include("build_max_z.jl")
include("find_initial_feasible_scenario_master.jl")

function test_build_max_z(; data_path=nothing, is_silent=false, sigma_star=0.0, solve_model=true)
    if data_path === nothing
        _, _, uncertainty_set, control_set, grid_set, state_set, microgrid_params = read_microgrid_problem_data()
    else
        _, _, uncertainty_set, control_set, grid_set, state_set, microgrid_params = read_microgrid_problem_data(data_path)
    end

    P = uncertainty_set["P"]
    p = uncertainty_set["p"]
    H = grid_set["H"]
    h = grid_set["h"]

    grid_size = microgrid_params["grid_size"]
    soc0 = microgrid_params["SoC0"]
    soc_min = microgrid_params["SOC_MIN"]
    soc_max = microgrid_params["SOC_MAX"]
    pc_max = microgrid_params["Pc_MAX"]
    pd_max = microgrid_params["Pd_MAX"]
    battery_cost = get(microgrid_params, "battery_cost", 0.0)
    delta_t = microgrid_params["Δt"]
    e_batt = microgrid_params["E_BATT"]

    x_blocks = Vector{Vector{VariableRef}}(undef, 1)
    z_blocks = Vector{Vector{VariableRef}}(undef, 1)
    l_blocks = Vector{Vector{VariableRef}}(undef, 1)

    w_init = find_initial_feasible_scenario_master(P, p; is_silent=is_silent)

    master_model, x_blocks, z_blocks, l_blocks = build_master_problem(
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
        delta_t,
        e_batt,
        is_silent;
        w_init=w_init,
    )

    optimize!(master_model)

    u_star = value.(master_model[:u])
    gamma_star = value(master_model[:gamma])

    max_z_model, metadata = build_max_z(
        H,
        h,
        grid_size,
        u_star,
        w_init,
        gamma_star,
        sigma_star,
        is_silent,
    )

    println("Built build_max_z successfully.")
    println("logic row count = ", metadata[:n_logic_rows], " (expected ", 2 * (grid_size - 1), ")")
    println("balance row count = ", metadata[:n_balance_rows], " (expected ", 2 * (grid_size - 1), ")")
    println("epigraph row count = ", metadata[:n_epigraph_rows], " (expected 1)")

    if solve_model
        optimize!(max_z_model)
        println("termination_status = ", termination_status(max_z_model))
        println("primal_status = ", primal_status(max_z_model))
        if has_values(max_z_model)
            println("sigma_bar = ", objective_value(max_z_model))
        end
    end

    return max_z_model, metadata
end

test_build_max_z()
