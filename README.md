# Local Reduction for Robust Microgrid Sizing and Scheduling

This folder contains the Julia implementation used for the final IFAC paper:

**An Efficient Method for the Optimal Control of Microgrids Under Uncertainties using Local Reduction**

The code solves a robust microgrid sizing and power scheduling problem under uncertainty in load demand, PV generation, electricity prices, and battery efficiencies. Two formulations are implemented and compared:

- **MILP/MIP formulation:** binary variables and big-M constraints encode battery and grid import/export logic.
- **NLP formulation:** continuous auxiliary variables encode the same logic through an exact smooth, non-convex reformulation.

Both formulations are solved using a nested local reduction algorithm and evaluated using Monte Carlo simulations.

## Main Scripts

- `SolveLocalReduction.jl`
  Convenience script for running one local reduction instance and plotting trajectories.

- `run_local_red.jl`
  Main one-battery local reduction implementation used by the current result files. This builds the master problem, calls the nested `z` local reduction, adds violating uncertainty scenarios, and optionally saves `.jld2` results.

- `MonteCarloFromResult.jl`
  Runs Monte Carlo validation over saved MIP or NLP result files. It evaluates 100,000 uniformly sampled scenarios per valid result file by default and reports aggregate feasibility and violation statistics.

- `PlotMILPExperimentHeatmaps.jl`
  Generates side-by-side MILP and NLP heatmaps over initial SoC and grid-cap factor. Missing or failed result files are shown as `INF`.

- `RunLocalReductionExperiments.jl`
  Older batch runner for experiment sweeps. Some paths in this script target the older nonlinear-existence implementation and may not match the final one-battery `run_local_red.jl` workflow.

## Core Local Reduction Files

- `build_master_problem.jl`
  Builds the finite-scenario outer master problem.

- `add_master_scenario.jl`
  Adds a new uncertainty scenario to the outer master problem.

- `solve_local_red_z_lambda.jl`
  Runs the nested local reduction loop over sampled `(z, lambda)` scenarios for a fixed outer-master solution.

- `build_z_master.jl`
  Builds the finite-scenario `master_z` problem, which searches for a violating uncertainty scenario `w`.

- `add_master_z_scenario.jl`
  Adds sampled `(z, lambda)` scenarios to `master_z` and builds the selector/disjunction constraints.

- `build_max_z.jl`
  Builds the lower-level maximization problem used to find a new violating `(z, lambda)` scenario for `master_z`.

- `solve_master_z_with_restarts.jl`
  Solves `master_z` with deterministic and random restarts. Additional recovery restarts are used when no feasible point is returned.

- `initialize_master_z.jl`, `initialize_new_master_scenario.jl`
  Warm-start helpers.

- `load_problem_data.jl`
  Loads and constructs the horizon-dependent microgrid data, uncertainty set, and model parameters.

## Microgrid Model

The current final model is a one-battery, grid-connected microgrid with PV sizing. The decision variables include:

```julia
n >= 1   # battery sizing multiplier / number of battery packs
s >= 1   # PV sizing multiplier
gamma    # epigraph upper bound on cost
```

For each time interval `t`, the open-loop battery controls are:

```julia
u[:, t] = [Pc_t, Pd_t]
```

where:

- `Pc_t` is battery charging power.
- `Pd_t` is battery discharging power.

The recourse/grid variables for each uncertainty scenario are:

```julia
z_t = [PG_plus_t, PG_minus_t]
```

where `PG_plus_t` is grid import and `PG_minus_t` is grid export.

The uncertainty vector per time interval is:

```julia
w_t = [PL_t, PR_t, C_buy_t, C_sell_t, eta_ch_t, rho_dis_t]
```

where `rho_dis_t` is the reciprocal discharge-efficiency term used in the state update.

## Dynamics and Balance

The discrete battery energy dynamics are:

```text
x[t+1] = x[t] + (eta_ch_t * Pc_t - rho_dis_t * Pd_t) * delta_t
```

with initial condition:

```text
x[1] = SoC0 * n * E_BATT
```

State-of-charge bounds are enforced as:

```text
SOC_MIN * n * E_BATT <= x[t] <= SOC_MAX * n * E_BATT
```

The outer master enforces the power balance for every added uncertainty scenario:

```text
PG_plus_t - PG_minus_t + Pd_t - Pc_t - PL_t + s * PR_t = 0
```

Equivalently:

```text
PG_plus_t - PG_minus_t = PL_t - s * PR_t - Pd_t + Pc_t
```

In the `g` functions used by the nested local reduction, the explicit grid variables are eliminated and the implied grid-net requirement is checked against the grid cap:

```text
required_grid_net_t = PL_t - s_star * PR_t - Pd_star_t + Pc_star_t
```

with residual rows:

```text
required_grid_net_t - PG_MAX <= 0
-required_grid_net_t - PG_MAX <= 0
```

These rows are built in `add_master_z_scenario.jl` inside `_build_master_z_selector_rows!`.

## Objective

The epigraph objective minimizes the investment cost plus operating cost:

```text
n * E_BATT * battery_cost
+ s * base_pv_capacity_kw * pv_cost
+ sum_t delta_t * (C_buy_t * PG_plus_t - C_sell_t * PG_minus_t)
<= gamma
```

The optimization objective is:

```text
min gamma
```

Cost and energy rows are scaled internally for numerical conditioning, but saved and reported values use the original units.

## Logic Formulations

### MILP / MIP

The MILP formulation uses binary variables to prevent simultaneous charge/discharge and simultaneous grid import/export.

Battery logic:

```text
0 <= Pc_t <= n_max * Pc_MAX * x1_t
0 <= Pd_t <= n_max * Pd_MAX * (1 - x1_t)
```

Grid logic:

```text
0 <= PG_plus_t  <= PG_MAX * x2_t
0 <= PG_minus_t <= PG_MAX * (1 - x2_t)
```

This yields finite MILP master, master-z, and max-z problems when `is_mip=true`.

### NLP

The NLP formulation uses continuous auxiliary variables and bilinear constraints:

```text
lambda_1_t + lambda_2_t = 1
lambda_3_t + lambda_4_t = 1
lambda_1_t * Pc_t + lambda_2_t * Pd_t <= 0
lambda_3_t * PG_plus_t + lambda_4_t * PG_minus_t <= 0
```

This gives a sequence of non-convex NLPs when `is_mip=false`.

## Nested Local Reduction Algorithm

The robust problem is an existence-constrained semi-infinite program. The code solves it by nested local reduction:

1. Start with an initial uncertainty scenario `w`.
2. Solve the outer master problem for the current finite uncertainty set.
3. Fix the current master solution `(u_star, n_star, s_star, gamma_star)`.
4. Solve `master_z` to search for a violating uncertainty scenario `w_star`.
5. Inside `master_z`, solve `max_z` to generate new violating `(z, lambda)` scenarios.
6. If the nested violation is larger than `tol_2`, add the new `(z, lambda)` scenario to `master_z`.
7. If the outer violation is larger than `tol_1`, add `w_star` to the outer master.
8. Stop when no violation is found or the scenario limits are reached.

For the NLP formulation, the local solvers are start-sensitive. The implementation includes random restarts for `master_z`, plus recovery restarts if the master or master-z fails to return a feasible point.

## Saved Result Files

Results are saved under:

```text
Revision/Results
```

Current result names follow:

```text
MIP_SOC020_H1_PG20_yyyy-mm-dd_HHMMSS.jld2
NLP_SOC020_H1_PG20_yyyy-mm-dd_HHMMSS.jld2
```

where:

- `MIP` / `NLP` is the formulation.
- `SOC020` means `SoC0 = 20%`.
- `H1` means one-day horizon.
- `PG20` means grid cap factor `g = 20%`.

Older names using `MILP_..._B1_...` are still supported by the heatmap script.

## Monte Carlo Validation

`MonteCarloFromResult.jl` validates saved results over a grid of:

```text
SoC0: 20%, 40%, 60%, 80%, 100%
Grid cap: 20%, 40%, 60%, 80%, 100%
```

For each valid result file, it samples 100,000 uncertainty scenarios uniformly from the box uncertainty set. Failed result files and missing result files are reported but do not contribute to aggregate averages.

A scenario is counted as infeasible if it violates at least one physical feasibility row:

- SoC bounds
- Battery/grid logic
- Grid-cap constraints

Cost violations are reported separately and contribute to constraint-violation statistics, but do not make a scenario physically infeasible.

Run from the `Revision` folder:

```powershell
$env:RUN_MONTE_CARLO_FROM_RESULT_SCRIPT="1"
julia MonteCarloFromResult.jl
```

or interactively:

```julia
include("MonteCarloFromResult.jl")
mc_summary = run_monte_carlo_sweep("NLP"; n_scenarios=100_000)
mc_summary = run_monte_carlo_sweep("MIP"; n_scenarios=100_000)
```

The aggregate summary reports:

- Average runtime and maximum runtime.
- Average `gamma_star`.
- Average feasible and infeasible scenario percentages.
- Average violated constraints as a percentage of total constraints.
- Breakdown of violated SoC, logic, grid-cap, and cost rows.
- Average maximum constraint violation.

## Heatmap Plot

`PlotMILPExperimentHeatmaps.jl` generates the side-by-side MILP/NLP sizing heatmap used in the paper.

Each grid cell corresponds to one `(SoC0, grid cap)` pair:

- Top half: optimal battery sizing `n_star`.
- Bottom half: optimal PV sizing `s_star`.
- Black `INF`: missing result file or saved result without a finite solution.

Run interactively:

```julia
include("PlotMILPExperimentHeatmaps.jl")
fig, axs, heatmap_data, saved_path = plot_milp_experiment_heatmaps()
```

or from PowerShell:

```powershell
$env:RUN_EXPERIMENT_HEATMAPS="1"
julia PlotMILPExperimentHeatmaps.jl
```

The figure is saved by default to:

```text
Revision/Plots/LocalReductionHeatmaps.png
```

## Running One Local Reduction Instance

From the `Revision` folder:

```julia
include("run_local_red.jl")

res = run_local_red(
    is_mip=false,
    horizon_days=1,
    soc0=0.4,
    grid_power_factor=0.2,
    k1_max=10,
    k2_max=10,
    z_master_restarts=5,
    save_results=true,
)
```

For the MILP formulation:

```julia
res = run_local_red(
    is_mip=true,
    mip_optimizer="Gurobi",
    horizon_days=1,
    soc0=0.4,
    grid_power_factor=0.2,
    k1_max=10,
    k2_max=10,
    save_results=true,
)
```

## Dependencies

The code uses:

- Julia
- JuMP
- Ipopt
- Gurobi
- HiGHS / SCIP in some helper paths
- JLD2
- PyPlot
- CSV/DataFrames through the data-loading path

Gurobi and HSL linear solvers may require local installation and licenses outside Julia's package manager.

If using a project environment:

```julia
import Pkg
Pkg.activate(".")
Pkg.instantiate()
```

or start Julia with:

```powershell
julia --project=.
```

## Citation

If you use this code in your work, please cite the following paper.

```bibtex
@article{Scaccia2026,
  title   = {{An Efficient Method for the Optimal Control of Microgrids Under Uncertainties using Local Reduction}},
  author  = {Scaccia, Edoardo and Kerrigan, Eric C. and Sadowska, Anna},
  journal = {arXiv preprint arXiv:2606.12345},
  year    = {2026},
  url     = {https://arxiv.org/abs/2606.12345}
}
```
