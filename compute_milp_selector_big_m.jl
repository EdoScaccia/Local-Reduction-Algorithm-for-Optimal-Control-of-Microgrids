function milp_selector_scenario_sigma_ub(g_ubs)
    isempty(g_ubs) && error("Cannot compute sigma upper bound from an empty g_ubs vector.")
    all(isfinite, g_ubs) || error("Cannot compute sigma upper bound with non-finite g_ubs.")
    return maximum(g_ubs)
end

function milp_selector_global_sigma_ub(scenario_sigma_ubs; aggregation::Symbol=:minimum)
    isempty(scenario_sigma_ubs) && error("Cannot compute global sigma upper bound without scenario bounds.")
    all(isfinite, scenario_sigma_ubs) || error("Cannot compute global sigma upper bound with non-finite scenario bounds.")

    if aggregation == :minimum
        return minimum(scenario_sigma_ubs)
    elseif aggregation == :maximum
        return maximum(scenario_sigma_ubs)
    else
        error("Unknown sigma bound aggregation `$aggregation`. Use :minimum or :maximum.")
    end
end

function milp_selector_big_m_values(g_lbs, sigma_ub)
    all(isfinite, g_lbs) || error("Cannot compute Big-M values with non-finite g_lbs.")
    isfinite(sigma_ub) || error("Cannot compute Big-M values with non-finite sigma_ub.")
    return max.(0.0, sigma_ub .- g_lbs)
end

