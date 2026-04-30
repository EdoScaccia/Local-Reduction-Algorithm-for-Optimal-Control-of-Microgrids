using JuMP
using HiGHS
using Gurobi
using SCIP

if !(@isdefined add_exact_robust_microgrid_constraints!)
    include("exact_robust_microgrid_constraints.jl")
end

function build_master_problem_milp(
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
    w_init=nothing,
    tol=1e-8,
    two_batt::Bool=true,
    is_cycle::Bool=false,
    n_max::Real=100,
    m_max::Real=100,
    s_max::Real=100,
)
    """
    Build the outer master NLP.
    """
    n_w = size(P, 2)
    cost_scale = max(abs(base_pv_capacity_kw * pv_cost), abs(e_batt * battery_cost), 1.0)
    energy_scale = max(abs(e_batt * max(n_max, m_max)), 1.0)

    # Check if an initial scenario is provided and if so, check it is feasible and has the correct dimension
    if w_init !== nothing
        if length(w_init) != n_w
            error("Initial scenario has wrong dimension.")
        end
        r = P * w_init .- p
        if any(r .> tol)
            error("Initial scenario is not feasible.")
        end
    end

    # Check if the input optimizer is valid and create the JuMP model
    if milp_optimizer == "Gurobi"
        model = Model(Gurobi.Optimizer)
    elseif milp_optimizer == "SCIP"
        model = Model(SCIP.Optimizer)
    else
        model = Model(HiGHS.Optimizer)
    end

    # Set the model to be silent if requested
    if is_silent
        set_silent(model)
    end

    # Create variables (special case for the test problem)
    @variable(model, n, lower_bound = 1, upper_bound = n_max, Int) # battery number
    if two_batt
        @variable(model, m, lower_bound = 1, upper_bound = m_max, Int) # second battery number
    end
    @variable(model, gamma)
    @variable(model, s, lower_bound = 1, upper_bound = s_max, Int) # multiplicative factor for PV uncertainty
    u_dim = two_batt ? 4 : 2
    x_dim = two_batt ? 2 * grid_size : grid_size
    l_dim_per_t = two_batt ? 3 : 2
    # Control Variable u := (Pc1, Pd1, Pc2, Pd2) or (Pc1, Pd1)
    @variable(model, u[1:u_dim, 1:grid_size-1], lower_bound = 0)

    # Variable which are used for each scenario
    x_blocks[1] = @variable(model, [1:x_dim])
    z_blocks[1] = @variable(model, [1: 2*(grid_size-1)], lower_bound = 0)
    l_blocks[1] = @variable(model, [1:l_dim_per_t * (grid_size - 1)], Bin)

    set_start_value(n, 1)
    two_batt && set_start_value(m, 1)
    set_start_value(s, 1)
    set_start_value(gamma, cost_scale)
    for t in 1:(grid_size - 1)
        for j in 1:u_dim
            set_start_value(u[j, t], 0.0)
        end
    end
    for t in 1:grid_size
        set_start_value(x_blocks[1][t], soc0 * e_batt)
        if two_batt
            set_start_value(x_blocks[1][grid_size + t], soc0 * e_batt)
        end
    end
    for t in 1:(grid_size - 1)
        z_idx = 2 * (t - 1)
        l_idx = l_dim_per_t * (t - 1)
        set_start_value(z_blocks[1][z_idx + 1], 0.0)
        set_start_value(z_blocks[1][z_idx + 2], 0.0)
        set_start_value(l_blocks[1][l_idx + 1], 0.0)
        set_start_value(l_blocks[1][l_idx + 2], 0.0)
        if two_batt
            set_start_value(l_blocks[1][l_idx + 3], 0.0)
        end
    end

    # Feasible set
    @constraint(model, H * vec(z_blocks[1]) .<= h)
    # @constraint(model, n >= m)

    @constraint(model, [t in 1:grid_size-1], u[1, t] <= n * pc_max)
    @constraint(model, [t in 1:grid_size-1], u[2, t] <= n * pd_max)
    @constraint(model, [t in 1:grid_size], x_blocks[1][t] >= soc_min * n * e_batt)
    @constraint(model, [t in 1:grid_size], x_blocks[1][t] <= soc_max * n * e_batt)

    if two_batt
        @constraint(model, [t in 1:grid_size-1], u[3, t] <= m * pc_max)
        @constraint(model, [t in 1:grid_size-1], u[4, t] <= m * pd_max)
        @constraint(model, [t in 1:grid_size], x_blocks[1][grid_size + t] >= soc_min * m * e_batt)
        @constraint(model, [t in 1:grid_size], x_blocks[1][grid_size + t] <= soc_max * m * e_batt)
    end

    # Initial conditions
    @constraint(model, x_blocks[1][1] == soc0 * n * e_batt)
    if is_cycle
        @constraint(model, x_blocks[1][grid_size] == soc0 * n * e_batt)
    end
    if two_batt
        @constraint(model, x_blocks[1][grid_size + 1] == soc0 * m * e_batt)
        if is_cycle
            @constraint(model, x_blocks[1][2 * grid_size] == soc0 * m * e_batt)
        end
    end

    # Constraints
    for t in 1:grid_size-1
        z_idx = 2 * (t - 1)
        l_idx = l_dim_per_t * (t - 1)

        # logic constraints
        @constraint(model, u[1, t] <= n_max * pc_max * l_blocks[1][l_idx + 1])
        @constraint(model, u[2, t] <= n_max * pd_max * (1 - l_blocks[1][l_idx + 1]))

        if two_batt
            @constraint(model, u[3, t] <= m_max * pc_max * l_blocks[1][l_idx + 2])
            @constraint(model, u[4, t] <= m_max * pd_max * (1 - l_blocks[1][l_idx + 2]))
            @constraint(model, z_blocks[1][z_idx + 1] <= pg_max * l_blocks[1][l_idx + 3])
            @constraint(model, z_blocks[1][z_idx + 2] <= pg_max * (1 - l_blocks[1][l_idx + 3]))
        else
            @constraint(model, z_blocks[1][z_idx + 1] <= pg_max * l_blocks[1][l_idx + 2])
            @constraint(model, z_blocks[1][z_idx + 2] <= pg_max * (1 - l_blocks[1][l_idx + 2]))
        end
        
        # power balance
        idx = 6 * (t - 1)
        PL_t = w_init[idx + 1]
        PR_t = w_init[idx + 2]
        eta_ch_t = w_init[idx + 5]
        rho_dis_t = w_init[idx + 6]
        batt2_net_t = two_batt ? u[4, t] - u[3, t] : 0.0
        @constraint(
            model,
            z_blocks[1][z_idx + 1] - z_blocks[1][z_idx + 2] + u[2, t] - u[1, t] + batt2_net_t - PL_t + s * PR_t == 0
        )

        # State evolution
        @constraint(
            model,
            x_blocks[1][t + 1] == x_blocks[1][t] + (eta_ch_t * u[1, t] - rho_dis_t * u[2, t]) * delta_t
        )
        if two_batt
            @constraint(
                model,
                x_blocks[1][grid_size + t + 1] == x_blocks[1][grid_size + t] + (eta_ch_t * u[3, t] - rho_dis_t * u[4, t]) * delta_t
            )
        end
    end
    # Epigraph constraint
    battery_count = n
    if two_batt
        battery_count += m
    end
    @constraint(
        model,
        (battery_count * e_batt * battery_cost + s * base_pv_capacity_kw * pv_cost +
        delta_t * sum(
            w_init[6 * (t - 1) + 3] * z_blocks[1][2 * (t - 1) + 1] -
            w_init[6 * (t - 1) + 4] * z_blocks[1][2 * (t - 1) + 2]
            for t in 1:grid_size-1
        ) - gamma) / cost_scale <= 0
    )

    add_exact_robust_microgrid_constraints!(
        model,
        u,
        gamma,
        n,
        two_batt ? m : nothing,
        s,
        P,
        p,
        H,
        h,
        grid_size,
        soc0,
        soc_min,
        soc_max,
        battery_cost,
        pv_cost,
        base_pv_capacity_kw,
        delta_t,
        e_batt,
        cost_scale,
        energy_scale;
        two_batt=two_batt,
        is_cycle=is_cycle,
    )

    # Objective is to minimize gamma
    @objective(model, Min, gamma / cost_scale)

    return model, x_blocks, z_blocks, l_blocks
end
