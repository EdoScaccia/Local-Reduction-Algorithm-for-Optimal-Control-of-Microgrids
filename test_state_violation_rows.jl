using Test

include("solve_local_red_z_lambda.jl")

@testset "state violations are local-reduction rows" begin
    grid_size = 2
    u_star = [0.0, 1.0]
    w_scenario = [0.0, 0.0, 0.0, 0.0, 1.0, 1.0]
    z_scenario = [0.0, 0.0]
    lambda_scenario = [0.5, 0.5, 0.5, 0.5]

    rows = _z_master_g_row_values(
        z_scenario,
        lambda_scenario,
        w_scenario,
        u_star,
        0.0,
        1.0,
        0.0,
        1.0,
        200.0,
        1500.0,
        90.84,
        1.0;
        soc0=0.2,
        soc_min=0.2,
        soc_max=1.0,
        delta_t=1.0,
        energy_scale=1.0,
        two_batt=false,
        is_cycle=true,
    )

    @test length(rows) == local_reduction_g_row_count(grid_size; two_batt=false, is_milp=false, is_cycle=true)
    @test 1.0 in rows
    @test maximum(rows) > 0.0
end
