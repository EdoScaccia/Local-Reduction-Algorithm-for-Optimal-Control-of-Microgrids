# Revision Local Reduction Scaffold

This folder contains the local-reduction implementation for the microgrid sizing problem with nonlinear existence variables.

## Main Entry Points

- `SolveLocalReduction.jl`
  Runs one local-reduction instance and plots the resulting scenarios/control.
- `RunLocalReductionExperiments.jl`
  Runs batches of local-reduction instances over selected `SoC0`, horizon, and grid-power-factor values. Failed experiments are skipped and not saved.
- `run_local_reduction_nonlinear_existence.jl`
  Main outer loop over uncertainty scenarios `w`.

## Core Files

- `build_master_problem.jl`
  Builds the outer master problem.
- `solve_local_red_z_lambda.jl`
  Runs the nested local reduction over `z` and lambda scenarios for a fixed outer-master solution.
- `build_z_master.jl`
  Builds the upper problem in the nested `z` local reduction.
- `build_max_z.jl`
  Builds the lower-level maximization problem used to find new `z`/lambda scenarios.
- `initialize_new_master_scenario.jl`
  Initializes a newly added outer-master scenario.
- `initialize_master_z.jl`
  Initializes the nested `master_z` model.
- `solve_master_z_with_restarts.jl`
  Solves `master_z` with deterministic warm start plus random restarts.
- `load_problem_data.jl`
  Builds the horizon-dependent data, uncertainty set, control/grid sets, and microgrid parameters.

## Current Microgrid Model

The current model has two independent battery packs and one shared grid connection/PV sizing variable.

Sizing variables:

```julia
n >= 1   # number of battery pack 1 units
m >= 1   # number of battery pack 2 units
s >= 1   # PV capacity multiplier
```

The objective epigraph uses:

```julia
(n + m) * battery_cost + s * base_pv_capacity_kw * pv_cost + grid_cost - gamma <= 0
```

Internally, the master objective and epigraph row are scaled by a cost scale

```julia
cost_scale = max(base_pv_capacity_kw * pv_cost, battery_cost, 1.0)
```

so the solver sees an objective of order 1 rather than order `1e5`. The reported `gamma_star` is still recovered from the unscaled `gamma` variable.

Because the two battery packs are physically identical, the master includes the symmetry-breaking constraint:

```julia
n >= m
```

This does not remove any achievable aggregate battery capacity; it only prevents Ipopt from exploring duplicate solutions with the pack labels swapped.

## Variable Layout

For a horizon with `n_t = grid_size - 1` intervals, the master control is now stacked as four controls per time step:

```julia
u[:, t] = [Pc1_t, Pd1_t, Pc2_t, Pd2_t]
```

Flattened histories therefore have length:

```julia
n_u = 4 * n_t
```

Each scenario state vector stores both battery energy trajectories in one vector:

```julia
x_block = [x1_1, ..., x1_grid_size,
           x2_1, ..., x2_grid_size]
```

so:

```julia
n_x = 2 * grid_size
```

The auxiliary grid-flow variable remains:

```julia
z_t = [PG_plus_t, PG_minus_t]
```

## Logic Variables

Each time step now has three simplex pairs:

```julia
l_t = [
    lambda_batt1_charge,
    lambda_batt1_discharge,
    lambda_grid_import,
    lambda_grid_export,
    lambda_batt2_charge,
    lambda_batt2_discharge,
]
```

Thus each scenario logic vector has length:

```julia
6 * n_t
```

The nested `z` local-reduction row count is:

```julia
n_rows_s = 5 * n_t + 1
```

corresponding to:

- 3 logic rows per interval: battery 1, grid, battery 2
- 2 power-balance rows per interval
- 1 epigraph row

## Power Balance

The per-interval power balance is:

```julia
PG_plus_t - PG_minus_t
+ Pd1_t - Pc1_t
+ Pd2_t - Pc2_t
- PL_t + s * PR_t == 0
```

This equation is used consistently in:

- `build_master_problem.jl`
- added outer-master scenarios in `run_local_reduction_nonlinear_existence.jl`
- `build_max_z.jl`
- `solve_local_red_z_lambda.jl`

## State Dynamics

Battery pack 1:

```julia
x1[t + 1] = x1[t] + (eta_ch_t * Pc1_t - Pd1_t / eta_dis_t) * delta_t
```

Battery pack 2:

```julia
x2[t + 1] = x2[t] + (eta_ch_t * Pc2_t - Pd2_t / eta_dis_t) * delta_t
```

Initial conditions:

```julia
x1[1] == soc0 * n * e_batt
x2[1] == soc0 * m * e_batt
```

The terminal equality constraints are currently commented out. If terminal SoC equality is needed, restore the corresponding constraints in:

- `build_master_problem.jl`
- the added-scenario block in `run_local_reduction_nonlinear_existence.jl`
- `build_z_master.jl`

## Reverting To One Battery Pack

To revert to the previous one-pack model, the main edits are:

1. Set `u[:, t] = [Pc_t, Pd_t]` and `n_u = 2 * n_t`.
2. Set `x_block = [x_1, ..., x_grid_size]` and `n_x = grid_size`.
3. Remove the sizing variable `m`.
4. Remove the second battery state bounds, initial condition, and dynamics.
5. Change the logic vector back to `4 * n_t` entries with only battery and grid simplex pairs.
6. Change `n_rows_s` in the nested z-master back to `4 * n_t + 1`.
7. Replace `(n + m) * battery_cost` with `n * battery_cost`.
8. Replace the power balance with:

```julia
PG_plus_t - PG_minus_t + Pd_t - Pc_t - PL_t + s * PR_t == 0
```

The files most likely to need edits for a one-pack revert are:

- `build_master_problem.jl`
- `run_local_reduction_nonlinear_existence.jl`
- `build_z_master.jl`
- `build_max_z.jl`
- `solve_local_red_z_lambda.jl`
- `initialize_master_z.jl`
- `initialize_new_master_scenario.jl`
- `find_initial_feasible_scenario_z_lambda.jl`

## Running

From the `Revision` folder, activate and instantiate the project environment:

```julia
import Pkg
Pkg.activate(".")
Pkg.instantiate()
```

or start Julia with:

```bash
julia --project=.
```

The project depends on JuMP, Ipopt, HiGHS, Gurobi, SCIP, HSL, JLD2, CSV/DataFrames, and PyPlot. Gurobi and HSL may also require local solver/license setup outside Julia's package manager.

Single run:

```julia
include("Revision/SolveLocalReduction.jl")
```

Direct function call:

```julia
include("Revision/run_local_reduction_nonlinear_existence.jl")
res = run_local_reduction_nonlinear_existence()
```

Experiment sweep:

```julia
include("Revision/RunLocalReductionExperiments.jl")
```

Successful run results are saved under `Revision/Results` when `save_results=true`.
