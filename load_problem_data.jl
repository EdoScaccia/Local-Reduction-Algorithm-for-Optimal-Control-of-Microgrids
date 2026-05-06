using CSV
using DataFrames
using Dates
using LinearAlgebra

function build_box_feasible_set(lb, ub, labels)
    """
    Build a box set in the form

        A * x - b <= 0

    from lower and upper bounds.
    """
    n = length(lb)
    A = vcat(Matrix{Float64}(I, n, n), -Matrix{Float64}(I, n, n))
    b = vcat(ub, -lb)
    return A, b, lb, ub, labels
end

function build_box_uncertainty_set(
    PL_lb,
    PL_ub,
    PR_lb,
    PR_ub,
    C_buy_lb,
    C_buy_ub,
    C_sell_lb,
    C_sell_ub,
    eta_ch_lb,
    eta_ch_ub,
    eta_dis_lb,
    eta_dis_ub,
)
    """
    Build the box uncertainty set

        P * w - p <= 0

    for the stacked vector

        w = [PL_1, PR_1, Cb_1, Cs_1, eta_ch_1, rho_dis_1, ..., PL_T, PR_T, Cb_T, Cs_T, eta_ch_T, rho_dis_T],
        where rho_dis_t = 1 / eta_dis_t.
    """
    n_t = length(PL_lb)
    n_w = 6 * n_t

    w_lb = zeros(n_w)
    w_ub = zeros(n_w)
    w_labels = Vector{String}(undef, n_w)

    for t in 1:n_t
        idx = 6 * (t - 1)

        w_lb[idx + 1] = PL_lb[t]
        w_lb[idx + 2] = PR_lb[t]
        w_lb[idx + 3] = C_buy_lb[t]
        w_lb[idx + 4] = C_sell_lb[t]
        w_lb[idx + 5] = eta_ch_lb[t]
        w_lb[idx + 6] = 1.0 / eta_dis_ub[t]

        w_ub[idx + 1] = PL_ub[t]
        w_ub[idx + 2] = PR_ub[t]
        w_ub[idx + 3] = C_buy_ub[t]
        w_ub[idx + 4] = C_sell_ub[t]
        w_ub[idx + 5] = eta_ch_ub[t]
        w_ub[idx + 6] = 1.0 / eta_dis_lb[t]

        w_labels[idx + 1] = "PL_$t"
        w_labels[idx + 2] = "PR_$t"
        w_labels[idx + 3] = "Cb_$t"
        w_labels[idx + 4] = "Cs_$t"
        w_labels[idx + 5] = "eta_ch_$t"
        w_labels[idx + 6] = "rho_dis_$t"
    end

    P, p, _, _, _ = build_box_feasible_set(w_lb, w_ub, w_labels)

    return P, p, w_lb, w_ub, w_labels
end

function build_control_feasible_set(grid_size, Pc_MAX, Pd_MAX)
    """
    Build the control feasible set

        U * u - q <= 0

    with

        u = [Pc_1, Pd_1, ..., Pc_T, Pd_T]

    and lower bounds equal to zero.
    """
    n_t = grid_size - 1
    n_u = 2 * n_t

    u_lb = zeros(n_u)
    u_ub = zeros(n_u)
    u_labels = Vector{String}(undef, n_u)

    for t in 1:n_t
        idx = 2 * (t - 1)
        u_ub[idx + 1] = Pc_MAX
        u_ub[idx + 2] = Pd_MAX
        u_labels[idx + 1] = "Pc_$t"
        u_labels[idx + 2] = "Pd_$t"
    end

    U, q, _, _, _ = build_box_feasible_set(u_lb, u_ub, u_labels)
    return U, q, u_lb, u_ub, u_labels
end

function build_grid_exchange_feasible_set(grid_size, PG_plus_MAX, PG_minus_MAX)
    """
    Build the auxiliary grid-flow feasible set

        H * z - h <= 0

    with

        z = [PG+_1, PG-_1, ..., PG+_T, PG-_T]

    and lower bounds equal to zero.
    """
    n_t = grid_size - 1
    n_z = 2 * n_t

    z_lb = zeros(n_z)
    z_ub = zeros(n_z)
    z_labels = Vector{String}(undef, n_z)

    for t in 1:n_t
        idx = 2 * (t - 1)
        z_ub[idx + 1] = PG_plus_MAX
        z_ub[idx + 2] = PG_minus_MAX
        z_labels[idx + 1] = "PG+_$t"
        z_labels[idx + 2] = "PG-_$t"
    end

    H, h, _, _, _ = build_box_feasible_set(z_lb, z_ub, z_labels)
    return H, h, z_lb, z_ub, z_labels
end

function build_state_feasible_set(grid_size, SOC_MIN, SOC_MAX)
    """
    Build the state feasible set

        G * x - g <= 0

    with

        SOC_MIN <= x_t <= SOC_MAX,  t = 1, ..., grid_size.
    """
    x_lb = fill(SOC_MIN, grid_size)
    x_ub = fill(SOC_MAX, grid_size)
    x_labels = ["x_$t" for t in 1:grid_size]

    G, g, _, _, _ = build_box_feasible_set(x_lb, x_ub, x_labels)
    return G, g, x_lb, x_ub, x_labels
end

function _parse_initial_day(x)
    if x isa Date
        return x
    elseif x isa DateTime
        return Date(x)
    elseif x isa AbstractString
        return try
            Date(x, dateformat"yyyy-mm-dd")
        catch
            Date(x, dateformat"dd-u-yyyy")
        end
    else
        error("Unsupported `initial_day` input: $x")
    end
end

function _repeat_to_length(v, n)
    m = length(v)
    m > 0 || error("Cannot repeat an empty vector.")
    return [v[mod1(i, m)] for i in 1:n]
end

function _build_prediction_bounds_from_processed_data(
    processed_data_path,
    initial_day,
    horizon_days,
    load_error_pct,
    pv_error_pct,
)
    df = CSV.read(processed_data_path, DataFrame)
    df.DateTime = DateTime.(df.DateTime, dateformat"dd-u-yyyy HH:MM:SS")

    day0 = _parse_initial_day(initial_day)
    t0 = DateTime(day0)
    t1 = t0 + Day(horizon_days)
    filtered = df[(df.DateTime .>= t0) .& (df.DateTime .< t1), :]
    nrow(filtered) > 0 || error("No processed data found between $t0 and $t1.")

    load_nominal = Float64.(filtered.Load)
    pv_nominal = Float64.(filtered.PVPower)

    load_lb = max.(0.0, load_nominal .* (1.0 - load_error_pct))
    load_ub = load_nominal .* (1.0 + load_error_pct)
    pv_lb = max.(0.0, pv_nominal .* (1.0 - pv_error_pct))
    pv_ub = pv_nominal .* (1.0 + pv_error_pct)

    return filtered.DateTime, load_nominal, pv_nominal, load_lb, load_ub, pv_lb, pv_ub
end

function read_microgrid_problem_data(
    data_path::AbstractString=normpath(joinpath(@__DIR__, "..", "Data.csv"));
    η_CH::Float64=0.95,
    η_DIS::Float64=0.91,
    η_uncertainty::Float64=0.15,
    SOC_MIN::Float64=0.2,
    SOC_MAX::Float64=1.0,
    Pc_MAX::Float64=90.0,
    Pd_MAX::Float64=90.0,
    E_BATT::Float64=326.0,
    PG_plus_MAX::Float64=1150.0,
    PG_minus_MAX::Float64=1150.0,
    SoC0::Float64=0.4,
    T::Int=24,
    Δt::Float64=0.25,
)
    """
    Read the microgrid CSV data and return:

    - `problem_data`
    - `uncertainty_bounds`
    - `uncertainty_set`
    - `control_set`
    - `grid_set`
    - `state_set`
    - `microgrid_params`

    The uncertainty set is built as

        P * w - p <= 0

    with

        w = [PL_1, PR_1, Cb_1, Cs_1, eta_ch_1, rho_dis_1, ..., PL_T, PR_T, Cb_T, Cs_T, eta_ch_T, rho_dis_T],
        where rho_dis_t = 1 / eta_dis_t.
    """
    problem_data = CSV.read(data_path, DataFrame; header=true)

    PR_ub = problem_data[:, 3]
    PR_lb = problem_data[:, 4]
    PL_ub = problem_data[:, 5]
    PL_lb = problem_data[:, 6]
    C_sell_ub = problem_data[:, 7]
    C_sell_lb = problem_data[:, 8]
    C_buy_ub = problem_data[:, 9]
    C_buy_lb = problem_data[:, 10]
    n_t = length(PL_lb)

    eta_ch_lb_scalar = max(0.0, η_CH * (1.0 - η_uncertainty))
    eta_ch_ub_scalar = min(1.0, η_CH * (1.0 + η_uncertainty))
    eta_dis_lb_scalar = max(0.0, η_DIS * (1.0 - η_uncertainty))
    eta_dis_ub_scalar = min(1.0, η_DIS * (1.0 + η_uncertainty))

    eta_ch_lb = fill(eta_ch_lb_scalar, n_t)
    eta_ch_ub = fill(eta_ch_ub_scalar, n_t)
    eta_dis_lb = fill(eta_dis_lb_scalar, n_t)
    eta_dis_ub = fill(eta_dis_ub_scalar, n_t)

    uncertainty_bounds = Dict(
        "PR_ub" => PR_ub,
        "PR_lb" => PR_lb,
        "PL_ub" => PL_ub,
        "PL_lb" => PL_lb,
        "C_sell_ub" => C_sell_ub,
        "C_sell_lb" => C_sell_lb,
        "C_buy_ub" => C_buy_ub,
        "C_buy_lb" => C_buy_lb,
        "eta_ch_ub" => eta_ch_ub,
        "eta_ch_lb" => eta_ch_lb,
        "eta_dis_ub" => eta_dis_ub,
        "eta_dis_lb" => eta_dis_lb,
        "rho_dis_ub" => 1.0 ./ eta_dis_lb,
        "rho_dis_lb" => 1.0 ./ eta_dis_ub,
    )

    P, p, w_lb, w_ub, w_labels = build_box_uncertainty_set(
        PL_lb,
        PL_ub,
        PR_lb,
        PR_ub,
        C_buy_lb,
        C_buy_ub,
        C_sell_lb,
        C_sell_ub,
        eta_ch_lb,
        eta_ch_ub,
        eta_dis_lb,
        eta_dis_ub,
    )

    uncertainty_set = Dict(
        "P" => P,
        "p" => p,
        "w_lb" => w_lb,
        "w_ub" => w_ub,
        "w_labels" => w_labels,
        "n_w" => length(w_lb),
        "n_t" => n_t
    )

    t = collect(0:Δt:T-Δt)
    grid_size = length(t)

    U, q, u_lb, u_ub, u_labels = build_control_feasible_set(grid_size, Pc_MAX, Pd_MAX)
    H, h, z_lb, z_ub, z_labels = build_grid_exchange_feasible_set(grid_size, PG_plus_MAX, PG_minus_MAX)
    G, g, x_lb, x_ub, x_labels = build_state_feasible_set(grid_size, SOC_MIN, SOC_MAX)

    control_set = Dict(
        "U" => U,
        "q" => q,
        "u_lb" => u_lb,
        "u_ub" => u_ub,
        "u_labels" => u_labels,
        "n_u" => length(u_lb)
    )

    grid_set = Dict(
        "H" => H,
        "h" => h,
        "z_lb" => z_lb,
        "z_ub" => z_ub,
        "z_labels" => z_labels,
        "n_z" => length(z_lb)
    )

    state_set = Dict(
        "G" => G,
        "g" => g,
        "x_lb" => x_lb,
        "x_ub" => x_ub,
        "x_labels" => x_labels,
        "n_x" => length(x_lb)
    )

    microgrid_params = Dict(
        "η_CH" => η_CH,
        "η_DIS" => η_DIS,
        "η_uncertainty" => η_uncertainty,
        "SOC_MIN" => SOC_MIN,
        "SOC_MAX" => SOC_MAX,
        "Pc_MAX" => Pc_MAX,
        "Pd_MAX" => Pd_MAX,
        "E_BATT" => E_BATT,
        "PG_plus_MAX" => PG_plus_MAX,
        "PG_minus_MAX" => PG_minus_MAX,
        "SoC0" => SoC0,
        "Δt" => Δt,
        "T" => T,
        "grid_size" => grid_size,
        "t" => t,
    )

    return problem_data, uncertainty_bounds, uncertainty_set, control_set, grid_set, state_set, microgrid_params
end

function load_problem_data(
    problem_source=nothing;
    processed_data_path::AbstractString=joinpath(@__DIR__, "Processed_Data_15min.csv"),
    horizon_days::Int=1,
    initial_day="2015-01-01",
    soc0::Float64=0.4,
    load_error_pct::Float64=0.10,
    pv_error_pct::Float64=0.10,
    pv_factor::Float64=1.0,
    grid_power_factor::Float64=1.0,
    battery_cost::Float64=200.0,
)
    if horizon_days < 1
        error("`horizon_days` must be at least 1.")
    end
    if load_error_pct < 0 || pv_error_pct < 0
        error("Prediction error percentages must be nonnegative.")
    end

    data_path = problem_source isa AbstractString ? problem_source : normpath(joinpath(@__DIR__, "..", "Data.csv"))
    problem_data, uncertainty_bounds, uncertainty_set, control_set, grid_set, state_set, microgrid_params =
        read_microgrid_problem_data(data_path; SoC0=soc0)

    timestamps, load_nominal, pv_nominal, load_lb, load_ub, pv_lb, pv_ub = _build_prediction_bounds_from_processed_data(
        processed_data_path,
        initial_day,
        horizon_days,
        load_error_pct,
        pv_error_pct,
    )
    n_t = length(load_lb)

    uncertainty_bounds["PL_lb"] = load_lb
    uncertainty_bounds["PL_ub"] = load_ub
    uncertainty_bounds["PR_lb"] = pv_factor .* pv_lb
    uncertainty_bounds["PR_ub"] = pv_factor .* pv_ub
    uncertainty_bounds["C_buy_lb"] = _repeat_to_length(uncertainty_bounds["C_buy_lb"], n_t)
    uncertainty_bounds["C_buy_ub"] = _repeat_to_length(uncertainty_bounds["C_buy_ub"], n_t)
    uncertainty_bounds["C_sell_lb"] = _repeat_to_length(uncertainty_bounds["C_sell_lb"], n_t)
    uncertainty_bounds["C_sell_ub"] = _repeat_to_length(uncertainty_bounds["C_sell_ub"], n_t)
    uncertainty_bounds["eta_ch_lb"] = _repeat_to_length(uncertainty_bounds["eta_ch_lb"], n_t)
    uncertainty_bounds["eta_ch_ub"] = _repeat_to_length(uncertainty_bounds["eta_ch_ub"], n_t)
    uncertainty_bounds["eta_dis_lb"] = _repeat_to_length(uncertainty_bounds["eta_dis_lb"], n_t)
    uncertainty_bounds["eta_dis_ub"] = _repeat_to_length(uncertainty_bounds["eta_dis_ub"], n_t)
    uncertainty_bounds["rho_dis_lb"] = 1.0 ./ uncertainty_bounds["eta_dis_ub"]
    uncertainty_bounds["rho_dis_ub"] = 1.0 ./ uncertainty_bounds["eta_dis_lb"]

    pg_max_from_data = grid_power_factor * maximum(uncertainty_bounds["PL_ub"])
    microgrid_params["PG_plus_MAX"] = pg_max_from_data
    microgrid_params["PG_minus_MAX"] = pg_max_from_data

    P, p, w_lb, w_ub, w_labels = build_box_uncertainty_set(
        uncertainty_bounds["PL_lb"],
        uncertainty_bounds["PL_ub"],
        uncertainty_bounds["PR_lb"],
        uncertainty_bounds["PR_ub"],
        uncertainty_bounds["C_buy_lb"],
        uncertainty_bounds["C_buy_ub"],
        uncertainty_bounds["C_sell_lb"],
        uncertainty_bounds["C_sell_ub"],
        uncertainty_bounds["eta_ch_lb"],
        uncertainty_bounds["eta_ch_ub"],
        uncertainty_bounds["eta_dis_lb"],
        uncertainty_bounds["eta_dis_ub"],
    )
    uncertainty_set = Dict(
        "P" => P,
        "p" => p,
        "w_lb" => w_lb,
        "w_ub" => w_ub,
        "w_labels" => w_labels,
        "n_w" => length(w_lb),
        "n_t" => n_t,
    )

    microgrid_params["battery_cost"] = battery_cost
    microgrid_params["grid_size"] = n_t + 1
    microgrid_params["T"] = n_t * microgrid_params["Δt"]
    microgrid_params["t"] = collect(0:microgrid_params["Δt"]:(n_t * microgrid_params["Δt"]))
    microgrid_params["prediction_timestamps"] = timestamps
    microgrid_params["load_nominal"] = load_nominal
    microgrid_params["pv_nominal"] = pv_nominal

    U, q, u_lb, u_ub, u_labels = build_control_feasible_set(
        microgrid_params["grid_size"],
        microgrid_params["Pc_MAX"],
        microgrid_params["Pd_MAX"],
    )
    control_set = Dict(
        "U" => U,
        "q" => q,
        "u_lb" => u_lb,
        "u_ub" => u_ub,
        "u_labels" => u_labels,
        "n_u" => length(u_lb),
    )

    H, h, z_lb, z_ub, z_labels = build_grid_exchange_feasible_set(
        microgrid_params["grid_size"],
        microgrid_params["PG_plus_MAX"],
        microgrid_params["PG_minus_MAX"],
    )
    grid_set = Dict(
        "H" => H,
        "h" => h,
        "z_lb" => z_lb,
        "z_ub" => z_ub,
        "z_labels" => z_labels,
        "n_z" => length(z_lb),
    )

    G, g, x_lb, x_ub, x_labels = build_state_feasible_set(
        microgrid_params["grid_size"],
        microgrid_params["SOC_MIN"],
        microgrid_params["SOC_MAX"],
    )
    state_set = Dict(
        "G" => G,
        "g" => g,
        "x_lb" => x_lb,
        "x_ub" => x_ub,
        "x_labels" => x_labels,
        "n_x" => length(x_lb),
    )

    return problem_data, uncertainty_bounds, uncertainty_set, control_set, grid_set, state_set, microgrid_params
end
