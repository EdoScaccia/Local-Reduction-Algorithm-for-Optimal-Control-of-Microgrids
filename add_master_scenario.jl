using JuMP

function add_master_scenario!(
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
    x_blocks[scenario_slot] = @variable(master_x, [1:x_dim_factor * grid_size])
    z_blocks[scenario_slot] = @variable(master_x, [1:2 * (grid_size - 1)], lower_bound = 0)
    l_dim_added = is_milp ? (two_batt ? 3 : 2) : (two_batt ? 6 : 4)
    l_blocks[scenario_slot] = is_milp ?
        @variable(master_x, [1:l_dim_added * (grid_size - 1)], Bin) :
        @variable(master_x, [1:l_dim_added * (grid_size - 1)], lower_bound=0)
    @constraint(master_x, H * vec(z_blocks[scenario_slot]) .<= h)

    if is_milp
        @constraint(master_x, [t in 1:grid_size], x_blocks[scenario_slot][t] >= soc_min * n * e_batt)
        @constraint(master_x, [t in 1:grid_size], x_blocks[scenario_slot][t] <= soc_max * n * e_batt)
    else
        @constraint(master_x, [t in 1:grid_size], (x_blocks[scenario_slot][t] - soc_min * n * e_batt) / energy_scale >= 0)
        @constraint(master_x, [t in 1:grid_size], (x_blocks[scenario_slot][t] - soc_max * n * e_batt) / energy_scale <= 0)
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

    if is_milp
        @constraint(master_x, x_blocks[scenario_slot][1] == soc0 * n * e_batt)
    else
        @constraint(master_x, (x_blocks[scenario_slot][1] - soc0 * n * e_batt) / energy_scale == 0)
    end
    if is_cycle
        if is_milp
            @constraint(master_x, x_blocks[scenario_slot][grid_size] == soc0 * n * e_batt)
        else
            @constraint(master_x, (x_blocks[scenario_slot][grid_size] - soc0 * n * e_batt) / energy_scale == 0)
        end
    end
    if two_batt
        m_var = master_x[:m]
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
            @constraint(master_x, l_blocks[scenario_slot][l_idx + 1] + l_blocks[scenario_slot][l_idx + 2] == 1)
            @constraint(master_x, l_blocks[scenario_slot][l_idx + 3] + l_blocks[scenario_slot][l_idx + 4] == 1)
            if two_batt
                @constraint(master_x, l_blocks[scenario_slot][l_idx + 5] + l_blocks[scenario_slot][l_idx + 6] == 1)
            end
            @constraint(master_x, l_blocks[scenario_slot][l_idx + 1] * u[1, t] + l_blocks[scenario_slot][l_idx + 2] * u[2, t] <= 0)
            @constraint(master_x, l_blocks[scenario_slot][l_idx + 3] * z_blocks[scenario_slot][z_idx + 1] + l_blocks[scenario_slot][l_idx + 4] * z_blocks[scenario_slot][z_idx + 2] <= 0)
            if two_batt
                @constraint(master_x, l_blocks[scenario_slot][l_idx + 5] * u[3, t] + l_blocks[scenario_slot][l_idx + 6] * u[4, t] <= 0)
            end
        end

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

        if is_milp
            @constraint(master_x, x_blocks[scenario_slot][t + 1] == x_blocks[scenario_slot][t] + (eta_ch_t * u[1, t] - rho_dis_t * u[2, t]) * delta_t)
        else
            @constraint(master_x, (x_blocks[scenario_slot][t + 1] - x_blocks[scenario_slot][t] - (eta_ch_t * u[1, t] - rho_dis_t * u[2, t]) * delta_t) / energy_scale == 0)
        end
        if two_batt
            if is_milp
                @constraint(master_x, x_blocks[scenario_slot][grid_size + t + 1] == x_blocks[scenario_slot][grid_size + t] + (eta_ch_t * u[3, t] - rho_dis_t * u[4, t]) * delta_t)
            else
                @constraint(master_x, (x_blocks[scenario_slot][grid_size + t + 1] - x_blocks[scenario_slot][grid_size + t] - (eta_ch_t * u[3, t] - rho_dis_t * u[4, t]) * delta_t) / energy_scale == 0)
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

    return nothing
end
