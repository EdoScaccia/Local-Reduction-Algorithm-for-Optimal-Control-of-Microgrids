using Dates
using JLD2
using PyPlot

mplcolors = PyPlot.matplotlib.colors
mplcm = PyPlot.matplotlib.cm
mplpatches = PyPlot.matplotlib.patches

results_dir = joinpath(@__DIR__, "Results")

soc0_values = [0.2, 0.4, 0.6, 0.8, 1.0]
grid_power_factor_values = [0.2, 0.4, 0.6, 0.8, 1.0]
horizon_days_values = [1, 3, 7]

is_milp = true
two_batt = false

function _pct_token(x)
    return round(Int, 100 * x)
end

function _battery_token(two_batt)
    return two_batt ? "B2" : "B1"
end

function _formulation_token(is_milp)
    return is_milp ? "MILP" : "NLP"
end

function _parse_result_filename(path)
    name = basename(path)
    m = match(r"^(MILP|NLP)_SOC0(\d+)_H(\d+)_PG(\d+)_(B[12])_(\d{4}-\d{2}-\d{2}_\d{6})\.jld2$", name)
    m === nothing && return nothing

    return (
        formulation=m.captures[1],
        soc0=parse(Int, m.captures[2]),
        horizon_days=parse(Int, m.captures[3]),
        grid_power_factor=parse(Int, m.captures[4]),
        battery=m.captures[5],
        timestamp=m.captures[6],
    )
end

function _latest_result_index(results_dir; is_milp=true, two_batt=false)
    isdir(results_dir) || error("Results directory does not exist: $results_dir")

    formulation = _formulation_token(is_milp)
    battery = _battery_token(two_batt)
    index = Dict{Tuple{Int, Int, Int}, String}()

    for path in readdir(results_dir; join=true)
        endswith(path, ".jld2") || continue
        cfg = _parse_result_filename(path)
        cfg === nothing && continue
        cfg.formulation == formulation || continue
        cfg.battery == battery || continue

        key = (cfg.soc0, cfg.horizon_days, cfg.grid_power_factor)
        if !haskey(index, key) || mtime(path) > mtime(index[key])
            index[key] = path
        end
    end

    return index
end

function _read_result_values(path)
    data = JLD2.load(path)
    _local_reduction_completed(data) || return NaN, NaN

    n_star = haskey(data, "n_star") ? Float64(data["n_star"]) : NaN
    s_star = haskey(data, "s_star") ? Float64(data["s_star"]) : NaN
    return n_star, s_star
end

function _local_reduction_completed(data)
    haskey(data, "scenarios_added") || return false
    return Int(data["scenarios_added"]) >= 5
end

function _build_heatmap_data(index, horizon_days; soc0_values, grid_power_factor_values)
    n_grid = fill(NaN, length(soc0_values), length(grid_power_factor_values))
    s_grid = fill(NaN, length(soc0_values), length(grid_power_factor_values))
    paths = Matrix{Union{Nothing, String}}(nothing, length(soc0_values), length(grid_power_factor_values))

    for (i, soc0) in enumerate(soc0_values)
        for (j, grid_power_factor) in enumerate(grid_power_factor_values)
            key = (_pct_token(soc0), horizon_days, _pct_token(grid_power_factor))
            haskey(index, key) || continue

            path = index[key]
            n_star, s_star = _read_result_values(path)
            if isfinite(n_star) && isfinite(s_star)
                n_grid[i, j] = n_star
                s_grid[i, j] = s_star
                paths[i, j] = path
            end
        end
    end

    return n_grid, s_grid, paths
end

function _default_n_star_bins(finite_n)
    upper = max(30.0, ceil(maximum(finite_n)))
    return [0.5, 1.5, 3.5, 5.5, 10.5, 15.5, 20.5, 25.5, upper + 0.5]
end

function _default_s_star_bins(finite_s)
    upper = max(30.0, ceil(maximum(finite_s)))
    return [0.5, 1.5, 3.5, 5.5, 10.5, 15.5, 20.5, 25.5, upper + 0.5]
end

function _bin_tick_labels(boundaries)
    labels = String[]
    for k in 1:(length(boundaries) - 1)
        lo = boundaries[k]
        hi = boundaries[k + 1]
        if k == 1
            push!(labels, "1")
        elseif k == length(boundaries) - 1
            push!(labels, ">$(round(Int, lo - 0.5))")
        else
            push!(labels, "$(round(Int, lo + 0.5))-$(round(Int, hi - 0.5))")
        end
    end
    return labels
end

function _n_colorbar_boundaries(boundaries)
    return vcat(boundaries[1] - 1.0, boundaries)
end

function _n_colorbar_labels(boundaries)
    return vcat(["INF"], _bin_tick_labels(boundaries))
end

function _n_colorbar_ticks(colorbar_boundaries)
    return [(colorbar_boundaries[k] + colorbar_boundaries[k + 1]) / 2 for k in 1:(length(colorbar_boundaries) - 1)]
end

function _value_color(value, cmap, norm)
    return cmap(norm(value))
end

function plot_milp_experiment_heatmaps(;
    results_dir=results_dir,
    soc0_values=soc0_values,
    grid_power_factor_values=grid_power_factor_values,
    horizon_days_values=horizon_days_values,
    is_milp=is_milp,
    two_batt=two_batt,
    n_star_bins=nothing,
    s_star_bins=nothing,
)
    index = _latest_result_index(results_dir; is_milp=is_milp, two_batt=two_batt)

    n_data = Matrix{Float64}[]
    s_data = Matrix{Float64}[]
    for horizon_days in horizon_days_values
        n_grid, s_grid, _ = _build_heatmap_data(
            index,
            horizon_days;
            soc0_values=soc0_values,
            grid_power_factor_values=grid_power_factor_values,
        )
        push!(n_data, n_grid)
        push!(s_data, s_grid)
    end

    finite_n = vcat([vec(grid[isfinite.(grid)]) for grid in n_data]...)
    isempty(finite_n) && error("No feasible result files found for the selected filters.")
    finite_s = vcat([vec(grid[isfinite.(grid)]) for grid in s_data]...)
    isempty(finite_s) && error("No finite s_star values found for the selected filters.")

    shared_values = vcat(finite_n, finite_s)
    shared_boundaries = if n_star_bins !== nothing
        collect(Float64, n_star_bins)
    elseif s_star_bins !== nothing
        collect(Float64, s_star_bins)
    else
        _default_n_star_bins(shared_values)
    end

    n_boundaries = shared_boundaries
    s_boundaries = shared_boundaries
    n_cbar_boundaries = _n_colorbar_boundaries(n_boundaries)

    cell_colors = [plt.cm.rainbow(x) for x in range(0.0, 1.0; length=length(shared_boundaries) - 1)]
    cbar_colors = vcat([[0.0, 0.0, 0.0, 1.0]], cell_colors)
    cell_cmap = mplcolors.ListedColormap(cell_colors)
    cbar_cmap = mplcolors.ListedColormap(cbar_colors)
    n_norm_for_cells = mplcolors.BoundaryNorm(n_boundaries, length(n_boundaries) - 1)
    n_norm_for_cbar = mplcolors.BoundaryNorm(n_cbar_boundaries, length(n_cbar_boundaries) - 1)
    s_norm = mplcolors.BoundaryNorm(s_boundaries, length(s_boundaries) - 1)

    fig, axs = plt.subplots(1, length(horizon_days_values), figsize=(15, 5), constrained_layout=true)
    if length(horizon_days_values) == 1
        axs = [axs]
    end

    for (plot_idx, horizon_days) in enumerate(horizon_days_values)
        ax = axs[plot_idx]
        n_grid = n_data[plot_idx]
        s_grid = s_data[plot_idx]

        ax.set_title("$(horizon_days)d")
        ax.set_xlabel("Grid Cap")
        plot_idx == 1 && ax.set_ylabel("Initial SoC")

        ax.set_xticks(0:length(grid_power_factor_values)-1)
        ax.set_yticks(0:length(soc0_values)-1)
        ax.set_xticklabels(["$(_pct_token(x))%" for x in grid_power_factor_values])
        ax.set_yticklabels(["$(_pct_token(x))%" for x in soc0_values])

        ax.set_xlim(-0.5, length(grid_power_factor_values) - 0.5)
        ax.set_ylim(-0.5, length(soc0_values) - 0.5)
        ax.set_aspect("equal")

        for i in 1:length(soc0_values)
            for j in 1:length(grid_power_factor_values)
                x0 = j - 1.5
                y0 = i - 1.5
                if isfinite(n_grid[i, j]) && isfinite(s_grid[i, j])
                    top_color = _value_color(n_grid[i, j], cell_cmap, n_norm_for_cells)
                    bottom_color = _value_color(s_grid[i, j], cell_cmap, s_norm)
                    ax.add_patch(mplpatches.Rectangle((x0, y0 + 0.5), 1.0, 0.5, facecolor=top_color, edgecolor="none"))
                    ax.add_patch(mplpatches.Rectangle((x0, y0), 1.0, 0.5, facecolor=bottom_color, edgecolor="none"))
                else
                    ax.add_patch(mplpatches.Rectangle((x0, y0), 1.0, 1.0, facecolor="black", edgecolor="none"))
                end
            end
        end

        for k in 0:length(grid_power_factor_values)
            ax.axvline(k - 0.5, color="white", linewidth=1.5)
        end
        for k in 0:length(soc0_values)
            ax.axhline(k - 0.5, color="white", linewidth=1.5)
        end
    end

    n_sm = mplcm.ScalarMappable(cmap=cbar_cmap, norm=n_norm_for_cbar)
    n_sm.set_array([])
    n_cbar = fig.colorbar(n_sm, ax=axs, ticks=_n_colorbar_ticks(n_cbar_boundaries), shrink=0.85)
    n_cbar.ax.set_yticklabels(_n_colorbar_labels(n_boundaries))
    n_cbar.set_label("Top half: n*, bottom half: s*")

    fig.suptitle("MILP Local Reduction Results", fontsize=14)

    display(fig)
    return fig, axs
end

fig, axs = plot_milp_experiment_heatmaps()
