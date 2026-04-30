function print_initial_summary(
    n_x,
    n_u,
    n_z,
    n_w,
    k1_max,
    k2_max,
    k3_max,
    tol_1,
    tol_2,
    processed_data_path,
    horizon_days,
    initial_day,
    soc0,
    load_error_pct,
    pv_error_pct,
    pv_factor,
    grid_power_factor,
    battery_factor,
    battery_cost,
    pv_cost,
    base_pv_capacity_kw
)
    println(repeat("=", 52))
    println("Problem dimensions:")
    println("n_x = ", n_x)
    println("n_u = ", n_u)
    println("n_z = ", n_z)
    println("n_w = ", n_w)
    println()
    println("Data parameters:")
    println("processed_data_path = ", processed_data_path)
    println("horizon_days = ", horizon_days)
    println("initial_day = ", initial_day)
    println("soc0 = ", soc0)
    println("load_error_pct = ", load_error_pct)
    println("pv_error_pct = ", pv_error_pct)
    println("pv_factor = ", pv_factor)
    println("grid_power_factor = ", grid_power_factor)
    println("battery_factor = ", battery_factor)
    println("battery_cost = ", battery_cost)
    println("pv_cost = ", pv_cost)
    println("base_pv_capacity_kw = ", base_pv_capacity_kw)
    println()
    println("Algorithm parameters:")
    println("k1_max = ", k1_max)
    println("k2_max = ", k2_max)
    println("k3_max = ", k3_max)
    println("tol_1 = ", tol_1)
    println("tol_2 = ", tol_2)
    println(repeat("=", 52))
end
