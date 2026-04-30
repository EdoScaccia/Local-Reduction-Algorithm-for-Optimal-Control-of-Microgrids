using JuMP
using MathOptInterface

function assert_solution!(
    model::JuMP.Model;
    msg_prefix::AbstractString = "No solution found.",
    acceptable_ts = (
        MathOptInterface.OPTIMAL,
        MathOptInterface.LOCALLY_SOLVED,
        MathOptInterface.ALMOST_OPTIMAL,
        MathOptInterface.ALMOST_LOCALLY_SOLVED,
    ),
    acceptable_ps = (
        MathOptInterface.FEASIBLE_POINT,
        MathOptInterface.NEARLY_FEASIBLE_POINT,
    ),
)
    """
    Check solver status and throw a custom error if no acceptable solution is available.
    """
    ts = termination_status(model)
    ps = primal_status(model)

    ok = (ts in acceptable_ts) && (ps in acceptable_ps)
    if !ok
        error("$msg_prefix - Termination status: $ts, primal status: $ps")
    end
    return nothing
end
