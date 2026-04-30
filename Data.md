# Data In The Local Reduction Algorithm

## Overview

The local reduction workflow in `Revision` uses two data sources:

1. A base CSV data file used to initialize price bounds, nominal battery parameters, and grid limits.
2. A preprocessed 15-minute time-series dataset used to define the prediction horizon for load and PV.

The key idea is:

- the processed 15-minute dataset provides the nominal forecast trajectories for `Load` and `PV`
- a slice of consecutive days is extracted from that dataset
- separate uncertainty bands are built around those nominal trajectories
- those bands become the uncertainty set used by the local reduction algorithm

## Main Files

- [Processed_Data_15min.csv](</c:/Users/es424/OneDrive - Imperial College London/Desktop/ICL/PhD/Julia/IFAC26/Revision/Processed_Data_15min.csv>)
- [load_problem_data.jl](</c:/Users/es424/OneDrive - Imperial College London/Desktop/ICL/PhD/Julia/IFAC26/Revision/load_problem_data.jl>)
- [run_local_reduction_nonlinear_existence.jl](</c:/Users/es424/OneDrive - Imperial College London/Desktop/ICL/PhD/Julia/IFAC26/Revision/run_local_reduction_nonlinear_existence.jl>)

## Source Data

### Base Data File

The low-level loader `read_microgrid_problem_data(...)` reads the base CSV file:

- by default: `../Data.csv`

From this file, the code extracts:

- `PR_ub`, `PR_lb`
- `PL_ub`, `PL_lb`
- `C_sell_ub`, `C_sell_lb`
- `C_buy_ub`, `C_buy_lb`

It also defines the nominal physical parameters:

- `η_CH`, `η_DIS`
- `SOC_MIN`, `SOC_MAX`
- `Pc_MAX`, `Pd_MAX`
- `E_BATT`
- `PG_plus_MAX`, `PG_minus_MAX`
- `SoC0`

and then builds the corresponding box sets for:

- uncertainty `w`
- control `u`
- grid flow `z`
- state `x`

### Processed 15-Minute Dataset

The higher-level loader `load_problem_data(...)` also reads:

- [Processed_Data_15min.csv](</c:/Users/es424/OneDrive - Imperial College London/Desktop/ICL/PhD/Julia/IFAC26/Revision/Processed_Data_15min.csv>)

This file is treated as the source of the nominal forecast trajectories.

Its main columns are:

- `DateTime`
- `PVPower`
- `Load`

The timestamps are parsed with:

```julia
dateformat"dd-u-yyyy HH:MM:SS"
```

Example:

```text
01-Jan-2015 00:00:00
```

## Horizon Selection

The higher-level data interface is:

```julia
load_problem_data(
    problem_source=nothing;
    processed_data_path=...,
    horizon_days=1,
    initial_day="2015-01-01",
    load_error_pct=0.10,
    pv_error_pct=0.10,
    pv_factor=1.0,
    battery_factor=1.0,
)
```

The horizon is chosen from:

- `initial_day`
- `horizon_days`

The loader computes:

- `t0 = DateTime(initial_day)`
- `t1 = t0 + Day(horizon_days)`

and extracts all rows satisfying:

```julia
t0 <= DateTime < t1
```

So the selected window is half-open:

- it includes the first timestamp of `initial_day`
- it excludes the first timestamp after the final day

If the processed data are 15-minute data, then:

- `1` day gives `96` time points
- `2` days gives `192` time points

and so on.

## Nominal Prediction

After slicing the processed dataset:

- `Load` is treated as the nominal load prediction
- `PVPower` is treated as the nominal PV prediction

In the code:

```julia
load_nominal = Float64.(filtered.Load)
pv_nominal = Float64.(filtered.PVPower)
```

These nominal trajectories are not used directly as deterministic signals.
Instead, they are converted into uncertainty bounds.

## Load And PV Bounds

The bounds are built using separate percentage errors:

- `load_error_pct`
- `pv_error_pct`

The code constructs:

```julia
load_lb = max.(0.0, load_nominal .* (1.0 - load_error_pct))
load_ub = load_nominal .* (1.0 + load_error_pct)

pv_lb = max.(0.0, pv_nominal .* (1.0 - pv_error_pct))
pv_ub = pv_nominal .* (1.0 + pv_error_pct)
```

So:

- load lower bound = nominal load minus a percentage error
- load upper bound = nominal load plus a percentage error
- PV lower bound = nominal PV minus a percentage error
- PV upper bound = nominal PV plus a percentage error

Both lower bounds are clipped at zero.

This means the processed dataset defines the center of the forecast, while the uncertainty percentages define the width of the band.

## PV Scaling

After the PV uncertainty band is built, an additional multiplicative factor is applied:

```julia
PR_lb = pv_factor .* pv_lb
PR_ub = pv_factor .* pv_ub
```

This is used to rescale the PV forecast envelope globally.

Interpretation:

- `pv_factor > 1` increases available PV
- `pv_factor < 1` decreases available PV

So the final PV uncertainty set is:

1. slice nominal PV from processed data
2. apply `pv_error_pct`
3. apply `pv_factor`

## Price And Efficiency Bounds

The processed dataset is currently used only for:

- load
- PV

The following are still taken from the base problem data:

- `C_buy_lb`, `C_buy_ub`
- `C_sell_lb`, `C_sell_ub`
- `η_CH` uncertainty bounds
- `η_DIS` uncertainty bounds

When the selected horizon length differs from the original base CSV horizon, these vectors are repeated to the required length.

In the code, the helper `_repeat_to_length(...)` is used for this purpose.

## Battery Scaling

The higher-level loader also applies a `battery_factor`:

```julia
Pc_MAX *= battery_factor
Pd_MAX *= battery_factor
E_BATT *= battery_factor
```

This changes:

- battery charge power limit
- battery discharge power limit
- battery energy capacity

These scaled values are then used to rebuild:

- the control feasible set
- the state dynamics scaling through `E_BATT`

## Uncertain Efficiencies

The algorithm treats charging and discharging efficiencies as uncertain.

The base loader first creates efficiency bounds from:

- nominal `η_CH`, `η_DIS`
- percentage uncertainty `η_uncertainty`

using:

```julia
η_CH * (1 ± η_uncertainty)
η_DIS * (1 ± η_uncertainty)
```

with clipping to `[0, 1]`.

These efficiency bounds are then repeated to match the selected prediction horizon if necessary.

## Uncertainty Vector

After all bounds are finalized, the uncertainty vector is built as:

```julia
w = [
    PL_1, PR_1, Cb_1, Cs_1, eta_ch_1, eta_dis_1,
    ...,
    PL_T, PR_T, Cb_T, Cs_T, eta_ch_T, eta_dis_T
]
```

where:

- `PL` = load
- `PR` = PV / renewable generation
- `Cb` = buying price
- `Cs` = selling price
- `eta_ch` = charging efficiency
- `eta_dis` = discharging efficiency

The uncertainty set is then written as a box:

```julia
P * w - p <= 0
```

This box is created by `build_box_uncertainty_set(...)`.

## Feasible Sets Built From The Data

Once the horizon and bounds are known, the loader rebuilds:

### Uncertainty set

```julia
P * w - p <= 0
```

### Control set

```julia
U * u - q <= 0
```

with

```julia
u = [Pc_1, Pd_1, ..., Pc_T, Pd_T]
```

### Grid-flow set

```julia
H * z - h <= 0
```

with

```julia
z = [PG+_1, PG-_1, ..., PG+_T, PG-_T]
```

### State set

```julia
G * x - g <= 0
```

with

```julia
SOC_MIN <= x_t <= SOC_MAX
```

## Time Grid

After the horizon length is determined from the processed data, the loader updates:

- `grid_size = n_t + 1`
- `T = n_t * Δt`
- `t = 0:Δt:n_t*Δt`

It also stores:

- `prediction_timestamps`

which are the actual timestamps from the extracted processed-data slice.

These timestamps are especially useful for plotting and post-processing.

## How The Algorithm Uses The Data

The local reduction algorithm uses the loaded data in this way:

1. `load_problem_data(...)` builds the horizon-specific problem.
2. The master problem uses:
   - the control set
   - the state set
   - one or more sampled scenarios from the uncertainty set
3. The `z`-master and lower-level problems use the same uncertainty structure to search for violating scenarios.
4. The plotting and Monte Carlo utilities read the saved or returned trajectories and interpret them consistently with the same data conventions.

## Practical Interpretation

The current workflow treats:

- processed historical data as a nominal prediction source
- uncertainty percentages as forecast error envelopes
- `pv_factor` as a scenario-wide PV rescaling
- `battery_factor` as a scenario-wide battery sizing rescaling

So each run is controlled by:

- what day window is selected
- how large the forecast error bands are
- how much PV is available
- how large the battery is

This makes it easy to run consistent experiments over different operating periods without rewriting the base model data.
