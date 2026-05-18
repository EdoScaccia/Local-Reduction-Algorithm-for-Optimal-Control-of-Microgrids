using JLD2
using PyPlot

mplcolors = PyPlot.matplotlib.colors
mplcm = PyPlot.matplotlib.cm
mplpatches = PyPlot.matplotlib.patches

results_dir = joinpath(@__DIR__, "Results")
plots_dir = joinpath(@__DIR__, "Plots")

soc0_values = [0.2, 0.4, 0.6, 0.8, 1.0]
grid_power_factor_values = [0.2, 0.4, 0.6, 0.8, 1.0]
horizon_days_values = [1]
formulations = ["MIP", "NLP"]

function _pct_token(x)
    return round(Int, 100 * x)
end

function _display_formulation(formulation)
    f = uppercase(String(formulation))
    return f in ("MIP", "MILP") ? "MILP" : f
end

function _canonical_formulation(formulation)
    f = uppercase(String(formulation))
    f in ("MIP", "MILP") && return "MIP"
    f == "NLP" && return "NLP"
    error("Unknown formulation `$formulation`. Use MIP/MILP or NLP.")
end

function _formulation_matches(file_formulation, requested_formulation)
    file_f = _canonical_formulation(file_formulation)
    requested_f = _canonical_formulation(requested_formulation)
    return file_f == requested_f
end

function _parse_result_filename(path)
    name = basename(path)
    m = match(
        r"^(MIP|MILP|NLP)_SOC0(\d+)_H(\d+)_PG(\d+)(?:_(B[12]))?_(\d{4}-\d{2}-\d{2}_\d{6})\.jld2$",
        name,
    )
    m === nothing && return nothing

    return (
        formulation=m.captures[1],
        soc0=parse(Int, m.captures[2]),
        horizon_days=parse(Int, m.captures[3]),
        grid_power_factor=parse(Int, m.captures[4]),
        battery=m.captures[5] === nothing ? "" : m.captures[5],
        timestamp=m.captures[6],
    )
end

function _latest_result_index(results_dir; formulations=formulations)
    isdir(results_dir) || error("Results directory does not exist: $results_dir")

    requested = Set(_canonical_formulation.(formulations))
    index = Dict{Tuple{String, Int, Int, Int}, String}()

    for path in readdir(results_dir; join=true)
        endswith(path, ".jld2") || continue
        cfg = _parse_result_filename(path)
        cfg === nothing && continue

        canonical = _canonical_formulation(cfg.formulation)
        canonical in requested || continue

        key = (canonical, cfg.soc0, cfg.horizon_days, cfg.grid_power_factor)
        if !haskey(index, key) || mtime(path) > mtime(index[key])
            index[key] = path
        end
    end

    return index
end

function _last_finite_col(A)
    for j in axes(A, 2)[end]:-1:axes(A, 2)[begin]
        if all(isfinite, A[:, j])
            return j
        end
    end
    return nothing
end

function _result_has_finite_solution(data)
    haskey(data, "n_star") && isfinite(Float64(data["n_star"])) || return false
    haskey(data, "s_star") && isfinite(Float64(data["s_star"])) || return false

    if haskey(data, "u_star_history")
        col = _last_finite_col(data["u_star_history"])
        col === nothing && return false
    end

    if haskey(data, "master_x_obj_history")
        any(isfinite, data["master_x_obj_history"]) || return false
    end

    return true
end

function _read_result_values(path)
    try
        data = JLD2.load(path)
        _result_has_finite_solution(data) || return NaN, NaN, :failed

        n_star = Float64(data["n_star"])
        s_star = Float64(data["s_star"])
        return n_star, s_star, :ok
    catch
        return NaN, NaN, :failed
    end
end

function _build_heatmap_data(index, formulation, horizon_days; soc0_values, grid_power_factor_values)
    n_grid = fill(NaN, length(soc0_values), length(grid_power_factor_values))
    s_grid = fill(NaN, length(soc0_values), length(grid_power_factor_values))
    status_grid = fill(:missing, length(soc0_values), length(grid_power_factor_values))
    paths = Matrix{Union{Nothing, String}}(nothing, length(soc0_values), length(grid_power_factor_values))

    canonical = _canonical_formulation(formulation)
    for (i, soc0) in enumerate(soc0_values)
        for (j, grid_power_factor) in enumerate(grid_power_factor_values)
            key = (canonical, _pct_token(soc0), horizon_days, _pct_token(grid_power_factor))
            if !haskey(index, key)
                status_grid[i, j] = :missing
                continue
            end

            path = index[key]
            n_star, s_star, status = _read_result_values(path)
            paths[i, j] = path
            status_grid[i, j] = status
            if status == :ok
                n_grid[i, j] = n_star
                s_grid[i, j] = s_star
            end
        end
    end

    return n_grid, s_grid, status_grid, paths
end

function _default_star_bins(finite_values)
    upper = max(30.0, ceil(maximum(finite_values)))
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

function _colorbar_boundaries(boundaries)
    return vcat(boundaries[1] - 1.0, boundaries)
end

function _colorbar_labels(boundaries)
    return vcat(["INF"], _bin_tick_labels(boundaries))
end

function _colorbar_ticks(colorbar_boundaries)
    return [(colorbar_boundaries[k] + colorbar_boundaries[k + 1]) / 2 for k in 1:(length(colorbar_boundaries) - 1)]
end

function _value_color(value, cmap, norm)
    return cmap(norm(value))
end

function _draw_heatmap_panel!(
    ax,
    n_grid,
    s_grid,
    status_grid,
    formulation,
    horizon_days,
    cell_cmap,
    cell_norm;
    soc0_values,
    grid_power_factor_values,
    show_ylabel=false,
)
    ax.set_title(_display_formulation(formulation))
    ax.set_xlabel("Grid Cap")
    show_ylabel && ax.set_ylabel("Initial SoC")

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
            if status_grid[i, j] == :ok && isfinite(n_grid[i, j]) && isfinite(s_grid[i, j])
                top_color = _value_color(n_grid[i, j], cell_cmap, cell_norm)
                bottom_color = _value_color(s_grid[i, j], cell_cmap, cell_norm)
                ax.add_patch(mplpatches.Rectangle((x0, y0 + 0.5), 1.0, 0.5, facecolor=top_color, edgecolor="none"))
                ax.add_patch(mplpatches.Rectangle((x0, y0), 1.0, 0.5, facecolor=bottom_color, edgecolor="none"))
            else
                ax.add_patch(mplpatches.Rectangle((x0, y0), 1.0, 1.0, facecolor="black", edgecolor="none"))
                ax.text(j - 1, i - 1, "INF", color="white", ha="center", va="center", fontsize=8, fontweight="bold")
            end
        end
    end

    for k in 0:length(grid_power_factor_values)
        ax.axvline(k - 0.5, color="white", linewidth=1.5)
    end
    for k in 0:length(soc0_values)
        ax.axhline(k - 0.5, color="white", linewidth=1.5)
    end

    return ax
end

function plot_milp_experiment_heatmaps(;
    results_dir=results_dir,
    plots_dir=plots_dir,
    soc0_values=soc0_values,
    grid_power_factor_values=grid_power_factor_values,
    horizon_days_values=horizon_days_values,
    formulations=formulations,
    star_bins=nothing,
    save_plot::Bool=true,
    filename::AbstractString="LocalReductionHeatmaps.png",
    dpi::Int=300,
)
    canonical_formulations = _canonical_formulation.(formulations)
    index = _latest_result_index(results_dir; formulations=canonical_formulations)

    panel_data = Dict{Tuple{String, Int}, NamedTuple}()
    finite_values = Float64[]

    for horizon_days in horizon_days_values
        for formulation in canonical_formulations
            n_grid, s_grid, status_grid, paths = _build_heatmap_data(
                index,
                formulation,
                horizon_days;
                soc0_values=soc0_values,
                grid_power_factor_values=grid_power_factor_values,
            )
            panel_data[(formulation, horizon_days)] = (
                n_grid=n_grid,
                s_grid=s_grid,
                status_grid=status_grid,
                paths=paths,
            )
            append!(finite_values, vec(n_grid[isfinite.(n_grid)]))
            append!(finite_values, vec(s_grid[isfinite.(s_grid)]))
        end
    end

    isempty(finite_values) && error("No finite result files found for the selected filters.")

    boundaries = star_bins === nothing ? _default_star_bins(finite_values) : collect(Float64, star_bins)
    cbar_boundaries = _colorbar_boundaries(boundaries)

    cell_colors = [plt.cm.rainbow(x) for x in range(0.0, 1.0; length=length(boundaries) - 1)]
    cbar_colors = vcat([[0.0, 0.0, 0.0, 1.0]], cell_colors)
    cell_cmap = mplcolors.ListedColormap(cell_colors)
    cbar_cmap = mplcolors.ListedColormap(cbar_colors)
    cell_norm = mplcolors.BoundaryNorm(boundaries, length(boundaries) - 1)
    cbar_norm = mplcolors.BoundaryNorm(cbar_boundaries, length(cbar_boundaries) - 1)

    n_rows = length(horizon_days_values)
    n_cols = length(canonical_formulations)
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4.8 * n_rows), squeeze=false, constrained_layout=true)

    for (row, horizon_days) in enumerate(horizon_days_values)
        for (col, formulation) in enumerate(canonical_formulations)
            data = panel_data[(formulation, horizon_days)]
            _draw_heatmap_panel!(
                axs[row, col],
                data.n_grid,
                data.s_grid,
                data.status_grid,
                formulation,
                horizon_days,
                cell_cmap,
                cell_norm;
                soc0_values=soc0_values,
                grid_power_factor_values=grid_power_factor_values,
                show_ylabel=col == 1,
            )
        end
    end

    sm = mplcm.ScalarMappable(cmap=cbar_cmap, norm=cbar_norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=vec(axs), ticks=_colorbar_ticks(cbar_boundaries), shrink=0.85)
    cbar.ax.set_yticklabels(_colorbar_labels(boundaries))
    cbar.set_label("Top half: b*, bottom half: s*")

    fig.suptitle("Local Reduction Result Heatmaps", fontsize=14)
    saved_path = nothing
    if save_plot
        mkpath(plots_dir)
        saved_path = joinpath(plots_dir, filename)
        fig.savefig(saved_path, dpi=dpi, bbox_inches="tight")
        println("Saved heatmap figure to: ", saved_path)
    end

    display(fig)
    return fig, axs, panel_data, saved_path
end

# if get(ENV, "RUN_EXPERIMENT_HEATMAPS", "0") == "1"
#     fig, axs, heatmap_data, saved_path = plot_milp_experiment_heatmaps()
# end
fig, axs, heatmap_data, saved_path = plot_milp_experiment_heatmaps()
