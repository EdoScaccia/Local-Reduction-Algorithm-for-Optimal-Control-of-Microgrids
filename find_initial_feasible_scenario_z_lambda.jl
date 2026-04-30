using JuMP
using LinearAlgebra
using HiGHS

include("assert_solution.jl")

function find_initial_feasible_scenario_z_lambda(H,
    h,
    grid_size,
    is_silent)
    """
    Solve a feasibility LP to find an initial `z` and `lambda`.
    """
    n_t = grid_size - 1

    model = Model(HiGHS.Optimizer)
    if is_silent
        set_silent(model)
    end

    @variable(model, z[1:2, 1:n_t] >= 0)
    @variable(model, lambda[1:6, 1:n_t] >= 0)

    @constraint(model, H * vec(z) .<= h)
    for t in 1:n_t
        # Simplex constraints on lambda
        @constraint(model, lambda[1, t] + lambda[2, t] == 1)
        @constraint(model, lambda[3, t] + lambda[4, t] == 1)
        @constraint(model, lambda[5, t] + lambda[6, t] == 1)
    end
    @objective(model, Min, 0)
    optimize!(model)

    assert_solution!(model; msg_prefix="No initial feasible solution found for Master Y.")
    return vec(value.(z)), vec(value.(lambda))
end
