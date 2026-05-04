using JuMP
using Ipopt

function build_master_problem(
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
    model = Model(Ipopt.Optimizer)
    set_attribute(model, "linear_solver", ipopt_lin_solver)
    set_attribute(model, "max_iter", max_iter_ipopt)
    set_attribute(model, "nlp_scaling_method", "gradient-based")
    set_attribute(model, "mu_strategy", "adaptive")
    set_attribute(model, "bound_push", 1e-8)
    set_attribute(model, "bound_frac", 1e-8)
    set_attribute(model, "constr_viol_tol", 1e-7)
    set_attribute(model, "acceptable_tol", 1e-5)
    set_attribute(model, "acceptable_constr_viol_tol", 1e-6)
    # Set the model to be silent if requested
    if is_silent
        set_silent(model)
    end

    # Create variables (special case for the test problem)
    @variable(model, n, lower_bound = 1, upper_bound = n_max) # battery number
    if two_batt
        @variable(model, m, lower_bound = 1, upper_bound = m_max) # second battery number
    end
    @variable(model, gamma)
    @variable(model, s, lower_bound = 1, upper_bound = s_max) # multiplicative factor for PV uncertainty
    u_dim = two_batt ? 4 : 2
    x_dim = two_batt ? 2 * grid_size : grid_size
    l_dim_per_t = two_batt ? 6 : 4
    # Control Variable u := (Pc1, Pd1, Pc2, Pd2) or (Pc1, Pd1)
    @variable(model, u[1:u_dim, 1:grid_size-1], lower_bound = 0)

    # Variable which are used for each scenario
    x_blocks[1] = @variable(model, [1:x_dim])
    z_blocks[1] = @variable(model, [1: 2*(grid_size-1)], lower_bound = 0)
    l_blocks[1] = @variable(model, [1:l_dim_per_t * (grid_size - 1)], lower_bound=0)

    set_start_value(n, 1.0)
    two_batt && set_start_value(m, 1.0)
    set_start_value(s, 1.0)
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
        set_start_value(l_blocks[1][l_idx + 1], 1e-3)
        set_start_value(l_blocks[1][l_idx + 2], 1.0 - 1e-3)
        set_start_value(l_blocks[1][l_idx + 3], 1e-3)
        set_start_value(l_blocks[1][l_idx + 4], 1.0 - 1e-3)
        if two_batt
            set_start_value(l_blocks[1][l_idx + 5], 1e-3)
            set_start_value(l_blocks[1][l_idx + 6], 1.0 - 1e-3)
        end
    end

    # Feasible set
    @constraint(model, H * vec(z_blocks[1]) .<= h)
    # @constraint(model, n >= m)

    @constraint(model, [t in 1:grid_size-1], u[1, t] <= n * pc_max)
    @constraint(model, [t in 1:grid_size-1], u[2, t] <= n * pd_max)
    @constraint(model, [t in 1:grid_size], (x_blocks[1][t] - soc_min * n * e_batt) / energy_scale >= 0)
    @constraint(model, [t in 1:grid_size], (x_blocks[1][t] - soc_max * n * e_batt) / energy_scale <= 0)

    if two_batt
        @constraint(model, [t in 1:grid_size-1], u[3, t] <= m * pc_max)
        @constraint(model, [t in 1:grid_size-1], u[4, t] <= m * pd_max)
        @constraint(model, [t in 1:grid_size], (x_blocks[1][grid_size + t] - soc_min * m * e_batt) / energy_scale >= 0)
        @constraint(model, [t in 1:grid_size], (x_blocks[1][grid_size + t] - soc_max * m * e_batt) / energy_scale <= 0)
    end

    # Initial conditions
    @constraint(model, (x_blocks[1][1] - soc0 * n * e_batt) / energy_scale == 0)
    if is_cycle
        @constraint(model, (x_blocks[1][grid_size] - soc0 * n * e_batt) / energy_scale == 0)
    end
    if two_batt
        @constraint(model, (x_blocks[1][grid_size + 1] - soc0 * m * e_batt) / energy_scale == 0)
        if is_cycle
            @constraint(model, (x_blocks[1][2 * grid_size] - soc0 * m * e_batt) / energy_scale == 0)
        end
    end

    # Constraints
    for t in 1:grid_size-1
        z_idx = 2 * (t - 1)
        l_idx = l_dim_per_t * (t - 1)

        # simplex constraints on lambda
        @constraint(model, l_blocks[1][l_idx + 1] + l_blocks[1][l_idx + 2] == 1)
        @constraint(model, l_blocks[1][l_idx + 3] + l_blocks[1][l_idx + 4] == 1)
        if two_batt
            @constraint(model, l_blocks[1][l_idx + 5] + l_blocks[1][l_idx + 6] == 1)
        end

        # logic constraints
        @constraint(model, l_blocks[1][l_idx + 1] * u[1, t] + l_blocks[1][l_idx + 2] * u[2, t] <= 0)
        @constraint(model, l_blocks[1][l_idx + 3] * z_blocks[1][z_idx + 1] + l_blocks[1][l_idx + 4] * z_blocks[1][z_idx + 2] <= 0)
        if two_batt
            @constraint(model, l_blocks[1][l_idx + 5] * u[3, t] + l_blocks[1][l_idx + 6] * u[4, t] <= 0)
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
            (x_blocks[1][t + 1] - x_blocks[1][t] - (eta_ch_t * u[1, t] - rho_dis_t * u[2, t]) * delta_t) / energy_scale == 0
        )
        if two_batt
            @constraint(
                model,
                (x_blocks[1][grid_size + t + 1] - x_blocks[1][grid_size + t] - (eta_ch_t * u[3, t] - rho_dis_t * u[4, t]) * delta_t) / energy_scale == 0
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
        sum(
            delta_t * (
                w_init[6 * (t - 1) + 3] * z_blocks[1][2 * (t - 1) + 1] -
                w_init[6 * (t - 1) + 4] * z_blocks[1][2 * (t - 1) + 2]
            )
            for t in 1:grid_size-1
        ) - gamma) / cost_scale <= 0
    )

    # Objective is to minimize gamma
    @objective(model, Min, gamma / cost_scale)

    return model, x_blocks, z_blocks, l_blocks
end
