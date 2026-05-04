using JuMP

function initialize_master_z!(
    master_z,
    s_blocks,
    z_scenarios_used::Int,
    P,
    p,
    u_star,
    soc0,
    n_star,
    m_star,
    delta_t,
    e_batt;
    w_start=nothing,
    x_witness=nothing,
    z_witness=nothing,
    sigma_start=0.0,
    s_start_vectors=nothing,
    two_batt::Bool=true,
)
    n_rows_w = size(P, 2)
    u_dim = two_batt ? 4 : 2
    x_dim_factor = two_batt ? 2 : 1
    n_t = div(length(u_star), u_dim)
    w_ub = p[1:n_rows_w]
    w_lb = -p[n_rows_w + 1:2 * n_rows_w]

    if w_start !== nothing && length(w_start) == n_rows_w
        w_init = clamp.(w_start, w_lb, w_ub)
    else
        w_init = 0.5 .* (w_lb .+ w_ub)
    end

    for i in 1:n_rows_w
        set_start_value(master_z[:w][i, 1], w_init[i])
    end

    if haskey(object_dictionary(master_z), :z)
        z_dim = length(master_z[:z])
        if z_witness !== nothing && length(z_witness) == z_dim
            for i in 1:z_dim
                set_start_value(master_z[:z][i], max(z_witness[i], 0.0))
            end
        elseif any(v -> start_value(v) === nothing, master_z[:z])
            for i in 1:z_dim
                set_start_value(master_z[:z][i], 0.0)
            end
        end
    end

    grid_size = n_t + 1
    x_init = zeros(x_dim_factor * grid_size)
    if x_witness !== nothing && length(x_witness) == length(x_init)
        x_init .= x_witness
    else
        x_init[1] = soc0 * n_star * e_batt
        if two_batt
            x_init[grid_size + 1] = soc0 * m_star * e_batt
        end
        for t in 1:n_t
            u_idx = u_dim * (t - 1)
            w_idx = 6 * (t - 1)
            eta_ch_t = w_init[w_idx + 5]
            rho_dis_t = w_init[w_idx + 6]
            x_init[t + 1] = x_init[t] + (eta_ch_t * u_star[u_idx + 1] - rho_dis_t * u_star[u_idx + 2]) * delta_t
            if two_batt
                x_init[grid_size + t + 1] = x_init[grid_size + t] + (eta_ch_t * u_star[u_idx + 3] - rho_dis_t * u_star[u_idx + 4]) * delta_t
            end
        end
    end

    for t in eachindex(x_init)
        set_start_value(master_z[:x][t], x_init[t])
    end

    set_start_value(master_z[:sigma], sigma_start)

    for k in 1:z_scenarios_used
        s_k = s_blocks[k]
        if s_start_vectors !== nothing && k <= length(s_start_vectors) && s_start_vectors[k] !== nothing
            s_start = s_start_vectors[k]
            for i in eachindex(s_k)
                set_start_value(s_k[i], s_start[i])
            end
        elseif any(v -> start_value(v) === nothing, s_k)
            s_start = 1.0 / length(s_k)
            for i in eachindex(s_k)
                set_start_value(s_k[i], s_start)
            end
        end
    end

    return nothing
end
