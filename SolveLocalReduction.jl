using Dates
using PyPlot

include("run_local_red.jl")

function _unpack_flat_w_scenario(w_flat)
    n_t = div(length(w_flat), 6)
    PL = [w_flat[6 * (t - 1) + 1] for t in 1:n_t]
    PR = [w_flat[6 * (t - 1) + 2] for t in 1:n_t]
    Cb = [w_flat[6 * (t - 1) + 3] for t in 1:n_t]
    Cs = [w_flat[6 * (t - 1) + 4] for t in 1:n_t]
    eta_ch = [w_flat[6 * (t - 1) + 5] for t in 1:n_t]
    eta_dis = [1.0 / w_flat[6 * (t - 1) + 6] for t in 1:n_t]
    return PL, PR, Cb, Cs, eta_ch, eta_dis
end

function _get_time_axes(microgrid_params, n_t)
    if haskey(microgrid_params, "prediction_timestamps")
        t_u = microgrid_params["prediction_timestamps"]
        t_soc = vcat(t_u[1] - Dates.Minute(15), t_u)
    else
        t = microgrid_params["t"]
        t_u = t[1:n_t]
        t_soc = t[1:n_t + 1]
    end
    return t_u, t_soc
end

function _plot_all_w_scenarios(res, uncertainty_bounds, microgrid_params)
    w_hist = res.w_scenarios_history
    n_t = div(size(w_hist, 1), 6)
    t_u, _ = _get_time_axes(microgrid_params, n_t)

    fig, axs = plt.subplots(3, 2, figsize=(14, 12), sharex=true)

    axs[1, 1].plot(t_u, uncertainty_bounds["PL_ub"], "r--", linewidth=2, label="PL ub/lb")
    axs[1, 1].plot(t_u, uncertainty_bounds["PL_lb"], "r--", linewidth=2)
    axs[1, 1].set_title("PL")
    axs[1, 1].grid(true, alpha=0.3)

    axs[1, 2].plot(t_u, uncertainty_bounds["PR_ub"], "r--", linewidth=2, label="PR ub/lb")
    axs[1, 2].plot(t_u, uncertainty_bounds["PR_lb"], "r--", linewidth=2)
    axs[1, 2].set_title("PR")
    axs[1, 2].grid(true, alpha=0.3)

    axs[2, 1].plot(t_u, uncertainty_bounds["C_buy_ub"], "r--", linewidth=2, label="Cb ub/lb")
    axs[2, 1].plot(t_u, uncertainty_bounds["C_buy_lb"], "r--", linewidth=2)
    axs[2, 1].set_title("C_buy")
    axs[2, 1].grid(true, alpha=0.3)

    axs[2, 2].plot(t_u, uncertainty_bounds["C_sell_ub"], "r--", linewidth=2, label="Cs ub/lb")
    axs[2, 2].plot(t_u, uncertainty_bounds["C_sell_lb"], "r--", linewidth=2)
    axs[2, 2].set_title("C_sell")
    axs[2, 2].grid(true, alpha=0.3)

    axs[3, 1].plot(t_u, uncertainty_bounds["eta_ch_ub"], "r--", linewidth=2, label="eta_ch ub/lb")
    axs[3, 1].plot(t_u, uncertainty_bounds["eta_ch_lb"], "r--", linewidth=2)
    axs[3, 1].set_title("eta_ch")
    axs[3, 1].grid(true, alpha=0.3)

    axs[3, 2].plot(t_u, uncertainty_bounds["eta_dis_ub"], "r--", linewidth=2, label="eta_dis ub/lb")
    axs[3, 2].plot(t_u, uncertainty_bounds["eta_dis_lb"], "r--", linewidth=2)
    axs[3, 2].set_title("eta_dis")
    axs[3, 2].grid(true, alpha=0.3)

    for k in 1:size(w_hist, 2)
        w_k = w_hist[:, k]
        all(isfinite, w_k) || continue
        PL, PR, Cb, Cs, eta_ch, eta_dis = _unpack_flat_w_scenario(w_k)
        scenario_label = k == 1 ? "Added scenarios" : nothing
        axs[1, 1].plot(t_u, PL, color="k", linewidth=1.5, alpha=0.6, label=scenario_label)
        axs[1, 2].plot(t_u, PR, color="k", linewidth=1.5, alpha=0.6, label=scenario_label)
        axs[2, 1].plot(t_u, Cb, color="k", linewidth=1.5, alpha=0.6, label=scenario_label)
        axs[2, 2].plot(t_u, Cs, color="k", linewidth=1.5, alpha=0.6, label=scenario_label)
        axs[3, 1].plot(t_u, eta_ch, color="k", linewidth=1.5, alpha=0.6, label=scenario_label)
        axs[3, 2].plot(t_u, eta_dis, color="k", linewidth=1.5, alpha=0.6, label=scenario_label)
    end

    for ax in axs
        ax.legend(loc="best")
    end

    axs[3, 1].set_xlabel("Time")
    axs[3, 2].set_xlabel("Time")

    fig.tight_layout()
    display(fig)
    return fig, axs
end

function _last_finite_scenario(w_scenarios_history)
    for j in size(w_scenarios_history, 2):-1:1
        if all(isfinite, w_scenarios_history[:, j])
            return w_scenarios_history[:, j]
        end
    end
    return w_scenarios_history[:, end]
end

function _reconstruct_soc_trajectories(u_star, w_scenarios_history, microgrid_params, n_star)
    n_t = div(size(w_scenarios_history, 1), 6)
    u_dim = div(length(u_star), n_t)
    two_batt = u_dim == 4
    energy1 = zeros(n_t + 1)
    energy2 = zeros(n_t + 1)
    soc1 = zeros(n_t + 1)
    soc2 = zeros(n_t + 1)
    w_last = _last_finite_scenario(w_scenarios_history)

    delta_t = microgrid_params["Δt"]
    e_batt = microgrid_params["E_BATT"]
    capacity1 = n_star * e_batt
    capacity2 = 1.0

    energy1[1] = microgrid_params["SoC0"] * capacity1
    energy2[1] = two_batt ? microgrid_params["SoC0"] * capacity2 : NaN
    soc1[1] = microgrid_params["SoC0"]
    soc2[1] = two_batt ? microgrid_params["SoC0"] : NaN

    for t in 1:n_t
        u_idx = u_dim * (t - 1)
        w_idx = 6 * (t - 1)
        eta_ch_t = w_last[w_idx + 5]
        rho_dis_t = w_last[w_idx + 6]
        energy1[t + 1] = energy1[t] + (eta_ch_t * u_star[u_idx + 1] - rho_dis_t * u_star[u_idx + 2]) * delta_t
        if two_batt
            energy2[t + 1] = energy2[t] + (eta_ch_t * u_star[u_idx + 3] - rho_dis_t * u_star[u_idx + 4]) * delta_t
        end
        soc1[t + 1] = energy1[t + 1] / capacity1
        soc2[t + 1] = two_batt ? energy2[t + 1] / capacity2 : NaN
    end

    return soc1, soc2
end

function _reconstruct_grid_flow_trajectory(u_star, w_scenarios_history, s_star)
    n_t = div(size(w_scenarios_history, 1), 6)
    u_dim = div(length(u_star), n_t)
    two_batt = u_dim == 4
    w_last = _last_finite_scenario(w_scenarios_history)
    PL, PR, _, _, _, _ = _unpack_flat_w_scenario(w_last)

    PG_plus = zeros(n_t)
    PG_minus = zeros(n_t)
    for t in 1:n_t
        u_idx = u_dim * (t - 1)
        rhs_t = u_star[u_idx + 1] + (two_batt ? u_star[u_idx + 3] : 0.0) + PL[t] - s_star * PR[t] - u_star[u_idx + 2] - (two_batt ? u_star[u_idx + 4] : 0.0)
        PG_plus[t] = max(rhs_t, 0.0)
        PG_minus[t] = max(-rhs_t, 0.0)
    end

    return PG_plus, PG_minus
end

function _plot_soc_and_control(res, microgrid_params)
    u_star = res.u_star
    soc1, soc2 = _reconstruct_soc_trajectories(u_star, res.w_scenarios_history, microgrid_params, res.n_star)
    PG_plus, PG_minus = _reconstruct_grid_flow_trajectory(u_star, res.w_scenarios_history, res.s_star)
    pg_plus_max = microgrid_params["PG_plus_MAX"]

    n_t = div(size(res.w_scenarios_history, 1), 6)
    u_dim = div(length(u_star), n_t)
    two_batt = u_dim == 4
    t_u, t_soc = _get_time_axes(microgrid_params, n_t)
    Pc1 = [u_star[u_dim * (k - 1) + 1] for k in 1:n_t]
    Pd1 = [u_star[u_dim * (k - 1) + 2] for k in 1:n_t]
    Pc2 = two_batt ? [u_star[u_dim * (k - 1) + 3] for k in 1:n_t] : zeros(n_t)
    Pd2 = two_batt ? [u_star[u_dim * (k - 1) + 4] for k in 1:n_t] : zeros(n_t)

    fig, axs = plt.subplots(3, 1, figsize=(12, 10), sharex=true)

    axs[1].plot(t_soc, soc1, "k-", linewidth=2, label="SoC pack 1")
    two_batt && axs[1].plot(t_soc, soc2, color="tab:purple", linewidth=2, label="SoC pack 2")
    axs[1].set_ylabel("SoC")
    axs[1].set_title("Final Control and Reconstructed SoC")
    axs[1].grid(true, alpha=0.3)
    axs[1].legend(loc="best")
    axs[1].ticklabel_format(axis="y", style="plain", useOffset=false)

    axs[2].step(t_u, Pc1, where="post", linewidth=2, color="tab:blue", label="Pc1")
    axs[2].step(t_u, Pd1, where="post", linewidth=2, color="tab:orange", label="Pd1")
    if two_batt
        axs[2].step(t_u, Pc2, where="post", linewidth=2, color="tab:cyan", label="Pc2")
        axs[2].step(t_u, Pd2, where="post", linewidth=2, color="tab:red", label="Pd2")
    end
    axs[2].set_ylabel("Battery Power")
    axs[2].grid(true, alpha=0.3)
    axs[2].legend(loc="best")
    axs[2].ticklabel_format(axis="y", style="plain", useOffset=false)

    axs[3].step(t_u, PG_plus, where="post", linewidth=2, color="tab:green", label="PG+")
    axs[3].step(t_u, PG_minus, where="post", linewidth=2, color="tab:red", label="PG-")
    axs[3].axhline(pg_plus_max, color="tab:green", linestyle="--", linewidth=1.5, alpha=0.8, label="PG+ max")
    axs[3].set_ylabel("Grid Power")
    axs[3].set_xlabel("Time")
    axs[3].grid(true, alpha=0.3)
    axs[3].legend(loc="best")

    fig.tight_layout()
    display(fig)
    return fig, axs
end

run_kwargs = (
    is_silent=true,
    ipopt_lin_solver="mumps",
    mip_optimizer="Gurobi",
    max_iter_ipopt=10000,
    save_results=true,
    is_mip=true,
    is_cycle=false,
    warm_start_from_1day=false,
    n_max=100,
    s_max=100,
    horizon_days=1,
    grid_power_factor=0.2,
    soc0=0.2,
    k1_max=100,
    k2_max=20,
    z_master_restarts=1,
)

res = run_local_red(; run_kwargs...)
_plot_all_w_scenarios(res, res.uncertainty_bounds, res.microgrid_params)
_plot_soc_and_control(res, res.microgrid_params)
