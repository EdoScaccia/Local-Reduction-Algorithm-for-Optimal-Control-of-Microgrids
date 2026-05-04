using Random

function randomize_master_z_lambda_starts!(s_blocks, rng, z_scenarios_used)
    """
    Randomize the start values for the lambda variables in the master_z model for each z scenario.
    """
    n_rows_s = size(s_blocks[1], 1)

    # For each scenario z attached to the model, generate a random lambda vector
    for col in 1:z_scenarios_used
        # Normalize the random vector to sum to 1 to satisfy the simplex constraint
        v = rand(rng, n_rows_s)
        v ./= sum(v)
        # Set the start values for the lambda variables in this column to the generated random vector
        for i in 1:n_rows_s
            set_start_value(s_blocks[col][i], v[i])
        end
    end

    return nothing
end

function solve_master_z_with_restarts!(master_z, s_blocks, z_scenarios_used, n_restarts, rng_seed, is_silent; solve_stats=nothing, solve_label="NLP")
    """
    Solve the master Z problem with multiple random restarts to try to find a better local optimum.
    The lambda vector is initilaized randomly for each restart, and the best solution across all restarts is returned.
    """
    # Use a seed for reproducibility
    rng = MersenneTwister(rng_seed)
    # Initialize the best solution tracking variables
    best_obj = -Inf
    best_w = nothing
    best_z = nothing
    best_x = nothing
    best_s_blocks = nothing
    best_sigma = nothing
    best_found = false

    # Iterate over the number of restarts. The first solve uses the deterministic
    # warm start already on the model; later solves perturb only the simplex
    # variables.
    restart_text = n_restarts == 1 ? "1 solve" : "$n_restarts random restarts"
    print("    ", "Solving $solve_label for $restart_text...")
    flush(stdout)
    for restart_id in 1:n_restarts
        if restart_id > 1
            randomize_master_z_lambda_starts!(s_blocks, rng, z_scenarios_used)
        end
        # Solve the master z problem
        solve_t0 = time()
        optimize!(master_z)
        solve_elapsed = time() - solve_t0
        if solve_stats !== nothing
            solve_stats[:nlp_count] += 1
            solve_stats[:nlp_time] += solve_elapsed
        end
        if restart_id < n_restarts
            print("$restart_id,")
        else
            println("$restart_id, Done.")
        end

        # Check the termination status
        ts = termination_status(master_z)
        ps = primal_status(master_z)
        ok = (ts in (
                JuMP.MOI.OPTIMAL,
                JuMP.MOI.LOCALLY_SOLVED,
                JuMP.MOI.ALMOST_OPTIMAL,
                JuMP.MOI.ALMOST_LOCALLY_SOLVED,
            )) && (ps in (
                JuMP.MOI.FEASIBLE_POINT,
                JuMP.MOI.NEARLY_FEASIBLE_POINT,
            ))

        # If the NLP solver returned a valid solution, check if it's the best one we've found across all restarts
        if ok
            obj = objective_value(master_z)
            if obj > best_obj
                best_obj = obj
                best_w = value.(master_z[:w])[:, 1]
                best_z = value.(master_z[:z])
                best_x = value.(master_z[:x])
                best_s_blocks = [value.(s_blocks[k]) for k in 1:z_scenarios_used]
                best_sigma = value(master_z[:sigma])
                best_found = true
            end
        end
    end

    # Throw an error if no valid solution was found across any of the restarts
    best_found || assert_solution!(master_z; msg_prefix="Master Z cannot be solved.")

    # Use the best solution found as a starting point for the next master solve after a new z scenario is added
    for i in axes(best_w, 1)
        set_start_value(master_z[:w][i, 1], best_w[i])
    end
    for i in eachindex(best_z)
        set_start_value(master_z[:z][i], best_z[i])
    end
    for k in 1:z_scenarios_used
        for i in eachindex(best_s_blocks[k])
            set_start_value(s_blocks[k][i], best_s_blocks[k][i])
        end
    end
    set_start_value(master_z[:sigma], best_sigma)

    return best_obj, best_w, best_z, best_x
end
