using JuMP
using PyPlot

include("load_problem_data.jl")
include("build_master_problem.jl")
include("find_initial_feasible_scenario_master.jl")

function _unpack_w_scenario(w, grid_size)
    n_t = grid_size - 1
    PL = [w[6 * (t - 1) + 1] for t in 1:n_t]
    PR = [w[6 * (t - 1) + 2] for t in 1:n_t]
    C_buy = [w[6 * (t - 1) + 3] for t in 1:n_t]
    C_sell = [w[6 * (t - 1) + 4] for t in 1:n_t]
    return PL, PR, C_buy, C_sell
end

function _plot_master_problem_solution(microgrid_params, w_init, x_block, u, z_block)
    grid_size = microgrid_params["grid_size"]
    Δt = microgrid_params["Δt"]
    t_state = collect(0:Δt:(grid_size - 1) * Δt)
    t_ctrl = t_state[1:end-1]

    x_val = value.(x_block)
    u_val = value.(u)
    z_val = value.(z_block)
    PL, PR, C_buy, C_sell = _unpack_w_scenario(w_init, grid_size)

    fig, axs = plt.subplots(4, 1, figsize=(16, 16), sharex=true)

    axs[1].plot(t_state, x_val, linewidth=2, color="black")
    axs[1].set_ylabel("State")
    axs[1].set_title("Initial Master Problem Solution")
    axs[1].grid(true, alpha=0.3)

    axs[2].step(t_ctrl, u_val[1, :], where="post", linewidth=2, color="tab:blue", label="Pc")
    axs[2].step(t_ctrl, u_val[2, :], where="post", linewidth=2, color="tab:orange", label="Pd")
    axs[2].set_ylabel("Control")
    axs[2].legend(loc="best")
    axs[2].grid(true, alpha=0.3)

    axs[3].step(t_ctrl, z_val[1, :], where="post", linewidth=2, color="tab:green", label="PG+")
    axs[3].step(t_ctrl, z_val[2, :], where="post", linewidth=2, color="tab:red", label="PG-")
    axs[3].set_ylabel("Grid Flow")
    axs[3].legend(loc="best")
    axs[3].grid(true, alpha=0.3)

    axs[4].step(t_ctrl, PL, where="post", linewidth=2, color="tab:purple", label="PL")
    axs[4].step(t_ctrl, PR, where="post", linewidth=2, color="tab:cyan", label="PR")
    axs[4].set_ylabel("PL / PR")
    axs[4].set_xlabel("Time")
    axs[4].grid(true, alpha=0.3)

    ax_price = axs[4].twinx()
    ax_price.plot(t_ctrl, C_buy, linewidth=2, linestyle="--", color="tab:brown", label="Cb")
    ax_price.plot(t_ctrl, C_sell, linewidth=2, linestyle="--", color="tab:gray", label="Cs")
    ax_price.set_ylabel("Prices")

    lines_left, labels_left = axs[4].get_legend_handles_labels()
    lines_right, labels_right = ax_price.get_legend_handles_labels()
    axs[4].legend(vcat(lines_left, lines_right), vcat(labels_left, labels_right), loc="best", ncol=4)

    fig.tight_layout()
    display(fig)
    return fig, axs
end

function test_master_problem_initial(; data_path=nothing, is_silent=false)
    if data_path === nothing
        problem_data, uncertainty_bounds, uncertainty_set, control_set, grid_set, state_set, microgrid_params = read_microgrid_problem_data()
    else
        problem_data, uncertainty_bounds, uncertainty_set, control_set, grid_set, state_set, microgrid_params = read_microgrid_problem_data(data_path)
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

    model, x_blocks, z_blocks, l_blocks = build_master_problem(
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

    optimize!(model)

    u = model[:u]
    gamma = model[:gamma]

    println("termination_status = ", termination_status(model))
    println("primal_status = ", primal_status(model))
    if has_values(model)
        println("objective(gamma) = ", objective_value(model))
        println("gamma = ", value(gamma))
        println("u[:,1] = ", value.(u[:, 1]))
        println("z[:,1] = ", value.(z_blocks[1][:, 1]))
        println("lambda[:,1] = ", value.(l_blocks[1][:, 1]))
        _plot_master_problem_solution(microgrid_params, w_init, x_blocks[1], u, z_blocks[1])
    end

    return model, w_init, x_blocks, z_blocks, l_blocks
end

test_master_problem_initial()
