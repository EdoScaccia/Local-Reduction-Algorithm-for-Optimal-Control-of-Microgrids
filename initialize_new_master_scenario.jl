using JuMP

function initialize_new_master_scenario!(
    model,
    x_block,
    z_block,
    l_block,
    u_star,
    w_star,
    gamma_star,
    n_star,
    s_star,
    soc0,
    delta_t,
    e_batt;
    x_witness=nothing,
    is_mip::Bool=false,
)
    u_dim = 2
    l_dim_per_t = is_mip ? 2 : 4
    n_t = div(length(u_star), u_dim)
    grid_size = n_t + 1

    haskey(object_dictionary(model), :gamma) && set_start_value(model[:gamma], gamma_star)

    for t in 1:n_t
        u_idx = u_dim * (t - 1)
        for j in 1:u_dim
            set_start_value(model[:u][j, t], u_star[u_idx + j])
        end
    end

    x_start = zeros(grid_size)
    if x_witness !== nothing && length(x_witness) == length(x_start)
        x_start .= x_witness
    else
        x_start[1] = soc0 * n_star * e_batt
    end
    z_start = zeros(2 * n_t)
    l_start = zeros(l_dim_per_t * n_t)

    for t in 1:n_t
        u_idx = u_dim * (t - 1)
        z_idx = 2 * (t - 1)
        l_idx = l_dim_per_t * (t - 1)
        w_idx = 6 * (t - 1)

        pc1_t = u_star[u_idx + 1]
        pd1_t = u_star[u_idx + 2]
        pl_t = w_star[w_idx + 1]
        pr_t = w_star[w_idx + 2]
        eta_ch_t = w_star[w_idx + 5]
        rho_dis_t = w_star[w_idx + 6]

        rhs_t = pc1_t + pl_t - s_star * pr_t - pd1_t
        pg_plus_t = max(rhs_t, 0.0)
        pg_minus_t = max(-rhs_t, 0.0)

        z_start[z_idx + 1] = pg_plus_t
        z_start[z_idx + 2] = pg_minus_t

        if x_witness === nothing
            x_start[t + 1] = x_start[t] + (eta_ch_t * pc1_t - rho_dis_t * pd1_t) * delta_t
        end

        if is_mip
            l_start[l_idx + 1] = pc1_t >= pd1_t ? 1.0 : 0.0
            l_start[l_idx + 2] = pg_plus_t >= pg_minus_t ? 1.0 : 0.0
        else
            if pc1_t >= pd1_t
                l_start[l_idx + 1] = 1e-3
                l_start[l_idx + 2] = 1.0 - 1e-3
            else
                l_start[l_idx + 1] = 1.0 - 1e-3
                l_start[l_idx + 2] = 1e-3
            end

            if pg_plus_t >= pg_minus_t
                l_start[l_idx + 3] = 1e-3
                l_start[l_idx + 4] = 1.0 - 1e-3
            else
                l_start[l_idx + 3] = 1.0 - 1e-3
                l_start[l_idx + 4] = 1e-3
            end
        end
    end

    haskey(object_dictionary(model), :n) && set_start_value(model[:n], n_star)
    haskey(object_dictionary(model), :s) && set_start_value(model[:s], s_star)

    for i in eachindex(x_block)
        set_start_value(x_block[i], x_start[i])
    end
    for i in eachindex(z_block)
        set_start_value(z_block[i], z_start[i])
    end
    for i in eachindex(l_block)
        set_start_value(l_block[i], l_start[i])
    end

    return nothing
end
