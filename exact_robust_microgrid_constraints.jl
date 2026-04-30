function _box_bounds_from_Pp(P, p)
    n = size(P, 2)
    size(P, 1) == 2n || error("Expected a box set with 2n rows.")
    return -p[n + 1:2n], p[1:n]
end

function add_exact_robust_microgrid_constraints!(
    model,
    u,
    gamma,
    n,
    m,
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
    two_batt::Bool=true,
    is_cycle::Bool=false,
)
    w_lb, w_ub = _box_bounds_from_Pp(P, p)
    n_t = grid_size - 1
    pg_plus_max = h[1]
    pg_minus_max = h[2]
    @variable(model, robust_period_cost[1:n_t])

    function battery_net(t)
        return u[1, t] - u[2, t] + (two_batt ? u[3, t] - u[4, t] : 0.0)
    end

    for t in 1:n_t
        w_idx = 6 * (t - 1)
        r_min = battery_net(t) + w_lb[w_idx + 1] - s * w_ub[w_idx + 2]
        r_max = battery_net(t) + w_ub[w_idx + 1] - s * w_lb[w_idx + 2]

        @constraint(model, r_max <= pg_plus_max)
        @constraint(model, r_min >= -pg_minus_max)

        # Worst-case period cost over the box occurs at the largest net grid
        # import/export requirement and the adverse price endpoint.
        @constraint(model, robust_period_cost[t] >= w_ub[w_idx + 3] * r_max)
        @constraint(model, robust_period_cost[t] >= w_lb[w_idx + 4] * r_max)
    end

    function add_battery_envelope!(pc_row, pd_row, units)
        cap = units * e_batt
        e_min = Vector{Any}(undef, grid_size)
        e_max = Vector{Any}(undef, grid_size)
        e_min[1] = soc0 * cap
        e_max[1] = soc0 * cap

        for t in 1:n_t
            w_idx = 6 * (t - 1)
            e_min[t + 1] = e_min[t] + (w_lb[w_idx + 5] * u[pc_row, t] - w_ub[w_idx + 6] * u[pd_row, t]) * delta_t
            e_max[t + 1] = e_max[t] + (w_ub[w_idx + 5] * u[pc_row, t] - w_lb[w_idx + 6] * u[pd_row, t]) * delta_t
        end

        for t in 1:grid_size
            @constraint(model, (e_min[t] - soc_min * cap) / energy_scale >= 0)
            @constraint(model, (e_max[t] - soc_max * cap) / energy_scale <= 0)
        end

        if is_cycle
            @constraint(model, (e_min[end] - soc0 * cap) / energy_scale >= 0)
            @constraint(model, (e_max[end] - soc0 * cap) / energy_scale <= 0)
        end
    end

    add_battery_envelope!(1, 2, n)
    if two_batt
        add_battery_envelope!(3, 4, m)
    end

    battery_count = n + (two_batt ? m : 0.0)
    @constraint(
        model,
        (
            battery_count * e_batt * battery_cost +
            s * base_pv_capacity_kw * pv_cost +
            delta_t * sum(robust_period_cost[t] for t in 1:n_t) -
            gamma
        ) / cost_scale <= 0
    )

    return nothing
end
