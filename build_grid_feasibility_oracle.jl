using JuMP
using HiGHS
using Gurobi
using SCIP

function build_grid_feasibility_oracle_models(
    P,
    p,
    grid_size,
    u_dim;
    milp_optimizer="HiGHS",
    is_silent::Bool=true,
    two_batt::Bool=true,
)
    n_w = size(P, 2)
    n_t = grid_size - 1
    oracles = Vector{NamedTuple{(:model, :t, :sign), Tuple{Model, Int, Float64}}}(undef, 2 * n_t)

    model_idx = 0
    for t in 1:n_t
        for sign in (1.0, -1.0)
            model_idx += 1
            if milp_optimizer == "Gurobi"
                model = Model(Gurobi.Optimizer)
            elseif milp_optimizer == "SCIP"
                model = Model(SCIP.Optimizer)
            else
                model = Model(HiGHS.Optimizer)
            end
            is_silent && set_silent(model)

            @variable(model, w[1:n_w])
            @constraint(model, P * w .<= p)

            @objective(model, Max, 0.0)

            oracles[model_idx] = (model=model, t=t, sign=sign)
        end
    end

    return oracles
end

function solve_grid_feasibility_oracle!(
    oracles,
    u_star,
    s_star,
    pg_max,
    two_batt::Bool=true,
)
    isempty(oracles) && error("Grid feasibility oracle has no models.")
    n_w = length(oracles[1].model[:w])
    u_dim = two_batt ? 4 : 2
    best_sigmas = fill(-Inf, 2)
    best_ws = zeros(n_w, 2)
    best_ts = JuMP.MOI.OPTIMIZE_NOT_CALLED
    best_ps = JuMP.MOI.NO_SOLUTION
    total_elapsed = 0.0
    n_solves = 0

    for oracle in oracles
        model = oracle.model
        t = oracle.t
        sign = oracle.sign
        side_idx = sign > 0 ? 1 : 2
        w = model[:w]
        w_idx = 6 * (t - 1)
        u_idx = u_dim * (t - 1)
        batt2_net_t = two_batt ? u_star[u_idx + 4] - u_star[u_idx + 3] : 0.0
        d_t = w[w_idx + 1] - s_star * w[w_idx + 2] -
              u_star[u_idx + 2] + u_star[u_idx + 1] - batt2_net_t
        @objective(model, Max, sign * d_t - pg_max)

        solve_t0 = time()
        optimize!(model)
        total_elapsed += time() - solve_t0
        n_solves += 1

        ts = termination_status(model)
        ps = primal_status(model)
        ok = (ts in (JuMP.MOI.OPTIMAL, JuMP.MOI.ALMOST_OPTIMAL)) &&
             (ps in (JuMP.MOI.FEASIBLE_POINT, JuMP.MOI.NEARLY_FEASIBLE_POINT))
        if ok
            sigma_candidate = objective_value(model)
            if sigma_candidate > best_sigmas[side_idx]
                best_sigmas[side_idx] = sigma_candidate
                best_ws[:, side_idx] .= value.(model[:w])
                best_ts = ts
                best_ps = ps
            end
        elseif all(isinf, best_sigmas)
            best_ts = ts
            best_ps = ps
        end
    end

    return best_sigmas, best_ws, total_elapsed, best_ts, best_ps, n_solves
end
