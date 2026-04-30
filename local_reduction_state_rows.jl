function local_reduction_state_row_count(grid_size; two_batt::Bool=true, is_cycle::Bool=false)
    n_batt = two_batt ? 2 : 1
    return 2 * n_batt * grid_size + (is_cycle ? 2 * n_batt : 0)
end

function local_reduction_g_row_count(grid_size; two_batt::Bool=true, is_milp::Bool=false, is_cycle::Bool=false)
    n_t = grid_size - 1
    logic_rows_per_t = is_milp ? (two_batt ? 12 : 8) : (two_batt ? 3 : 2)
    return (logic_rows_per_t + 2) * n_t + 1 +
           local_reduction_state_row_count(grid_size; two_batt=two_batt, is_cycle=is_cycle)
end

function fixed_dispatch_state_envelope(
    u_star,
    w_lb,
    w_ub,
    grid_size,
    soc0,
    n_star,
    m_star,
    delta_t,
    e_batt;
    two_batt::Bool=true,
)
    u_dim = two_batt ? 4 : 2
    n_t = grid_size - 1

    x1_min = Vector{Float64}(undef, grid_size)
    x1_max = Vector{Float64}(undef, grid_size)
    x2_min = two_batt ? Vector{Float64}(undef, grid_size) : Float64[]
    x2_max = two_batt ? Vector{Float64}(undef, grid_size) : Float64[]

    x1_min[1] = soc0 * n_star * e_batt
    x1_max[1] = x1_min[1]
    if two_batt
        x2_min[1] = soc0 * m_star * e_batt
        x2_max[1] = x2_min[1]
    end

    for t in 1:n_t
        w_idx = 6 * (t - 1)
        u_idx = u_dim * (t - 1)
        eta_lb = w_lb[w_idx + 5]
        eta_ub = w_ub[w_idx + 5]
        rho_lb = w_lb[w_idx + 6]
        rho_ub = w_ub[w_idx + 6]

        x1_min[t + 1] = x1_min[t] + (eta_lb * u_star[u_idx + 1] - rho_ub * u_star[u_idx + 2]) * delta_t
        x1_max[t + 1] = x1_max[t] + (eta_ub * u_star[u_idx + 1] - rho_lb * u_star[u_idx + 2]) * delta_t

        if two_batt
            x2_min[t + 1] = x2_min[t] + (eta_lb * u_star[u_idx + 3] - rho_ub * u_star[u_idx + 4]) * delta_t
            x2_max[t + 1] = x2_max[t] + (eta_ub * u_star[u_idx + 3] - rho_lb * u_star[u_idx + 4]) * delta_t
        end
    end

    return (x1_min=x1_min, x1_max=x1_max, x2_min=x2_min, x2_max=x2_max)
end
