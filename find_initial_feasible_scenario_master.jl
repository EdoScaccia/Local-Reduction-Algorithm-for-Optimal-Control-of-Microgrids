using JuMP
using HiGHS

include("assert_solution.jl")

function find_initial_feasible_scenario_master(
    P,
    p;
    is_silent=true,
)
    """
    Find an initial feasible uncertainty scenario for the outer master problem.
    """
    # Get the dimension of the uncertainty variables
    n_w = size(P, 2)

    # Check if the input optimizer is valid and create the JuMP model
    model_lp = Model(HiGHS.Optimizer)

    # Set the model to be silent if requested
    if is_silent
        set_silent(model_lp)
    end

    # Create variables, uncertainty constraints and dummy objective to find an initial feasible scenario
    @variable(model_lp, w[1:n_w])
    @constraint(model_lp, P * w .<= p)
    @objective(model_lp, Min, 0)
    optimize!(model_lp)

    # Check that we found a feasible solution
    assert_solution!(model_lp; msg_prefix="Cannot find an initial feasible scenario.")
    return w_initial = value.(w)
end
