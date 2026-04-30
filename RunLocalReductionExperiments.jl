include("run_local_reduction_nonlinear_existence.jl")

const DEFAULT_SOC0_VALUES = [0.2, 0.4, 0.6, 0.8]
const DEFAULT_HORIZON_DAYS_VALUES = [1, 3, 7, 14, 30]
const DEFAULT_GRID_POWER_FACTOR_VALUES = [0.2, 0.4, 0.6, 0.8, 1.0]

function _run_one_local_reduction_experiment!(
    experiment_records,
    experiment_set::AbstractString,
    experiment_index::Int,
    total_experiments::Int,
    problem_source,
    base_kwargs::NamedTuple,
    config::NamedTuple,
)
    println()
    println(repeat("=", 72))
    println("Experiment ", experiment_index, " / ", total_experiments, " | ", experiment_set)
    println("  soc0 = ", config.soc0)
    println("  horizon_days = ", config.horizon_days)
    println("  grid_power_factor = ", config.grid_power_factor)
    println(repeat("=", 72))
    flush(stdout)

    run_kwargs = merge(base_kwargs, config)
    t0 = time()
    elapsed = NaN

    try
        res = run_local_reduction_nonlinear_existence(problem_source; run_kwargs...)
        elapsed = time() - t0

        push!(experiment_records, Dict{Symbol, Any}(
            :status => "success",
            :experiment_set => String(experiment_set),
            :experiment_index => experiment_index,
            :soc0 => config.soc0,
            :horizon_days => config.horizon_days,
            :grid_power_factor => config.grid_power_factor,
            :elapsed_seconds => elapsed,
            :gamma_star => res.gamma_star,
            :n_star => res.n_star,
            :s_star => res.s_star,
            :scenarios_added => res.scenarios_added,
            :solve_stats => res.solve_stats,
            :results_path => res.results_path === nothing ? "" : String(res.results_path),
        ))

        println("Experiment succeeded in ", round(elapsed; digits=2), " s.")
    catch err
        elapsed = time() - t0
        error_message = sprint(showerror, err)

        push!(experiment_records, Dict{Symbol, Any}(
            :status => "failed",
            :experiment_set => String(experiment_set),
            :experiment_index => experiment_index,
            :soc0 => config.soc0,
            :horizon_days => config.horizon_days,
            :grid_power_factor => config.grid_power_factor,
            :elapsed_seconds => elapsed,
            :error_message => error_message,
        ))

        println("Experiment failed; continuing to the next one.")
        println("  Error: ", error_message)
    end
    flush(stdout)

    return experiment_records
end

function build_local_reduction_experiment_configs(;
    soc0_values=DEFAULT_SOC0_VALUES,
    horizon_days_values=DEFAULT_HORIZON_DAYS_VALUES,
    grid_power_factor_values=DEFAULT_GRID_POWER_FACTOR_VALUES,
)
    experiments = NamedTuple[]

    for soc0 in soc0_values
        for horizon_days in horizon_days_values
            push!(experiments, (
                experiment_set="soc0_horizon_sweep",
                soc0=Float64(soc0),
                horizon_days=Int(horizon_days),
                grid_power_factor=1.0,
            ))
        end
    end

    for soc0 in soc0_values
        for grid_power_factor in grid_power_factor_values
            push!(experiments, (
                experiment_set="soc0_grid_power_factor_sweep",
                soc0=Float64(soc0),
                horizon_days=1,
                grid_power_factor=Float64(grid_power_factor),
            ))
        end
    end

    return experiments
end

function run_local_reduction_experiments(
    problem_source=nothing;
    soc0_values=DEFAULT_SOC0_VALUES,
    horizon_days_values=DEFAULT_HORIZON_DAYS_VALUES,
    grid_power_factor_values=DEFAULT_GRID_POWER_FACTOR_VALUES,
    base_run_kwargs...,
)
    experiments = build_local_reduction_experiment_configs(
        soc0_values=soc0_values,
        horizon_days_values=horizon_days_values,
        grid_power_factor_values=grid_power_factor_values,
    )

    default_run_kwargs = (
        is_silent=true,
        ipopt_lin_solver="ma97",
        save_results=true,
        results_dir=joinpath(@__DIR__, "Results"),
    )
    base_kwargs = merge(default_run_kwargs, (; base_run_kwargs...))
    experiment_records = Vector{Dict{Symbol, Any}}()

    println("Starting ", length(experiments), " local reduction experiments.")
    println("Successful local reduction results will be saved to: ", String(base_kwargs.results_dir))

    for (idx, config) in enumerate(experiments)
        _run_one_local_reduction_experiment!(
            experiment_records,
            config.experiment_set,
            idx,
            length(experiments),
            problem_source,
            base_kwargs,
            (
                soc0=config.soc0,
                horizon_days=config.horizon_days,
                grid_power_factor=config.grid_power_factor,
            ),
        )
    end

    n_success = count(record -> record[:status] == "success", experiment_records)
    n_failed = length(experiment_records) - n_success

    println()
    println(repeat("=", 72))
    println("LOCAL REDUCTION EXPERIMENTS COMPLETE")
    println("  successful experiments = ", n_success)
    println("  failed experiments     = ", n_failed)
    println(repeat("=", 72))

    return (
        records=experiment_records,
        experiments=experiments,
    )
end

experiment_results = run_local_reduction_experiments()
