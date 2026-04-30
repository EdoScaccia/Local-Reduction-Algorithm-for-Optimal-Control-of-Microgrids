function print_final_summary(gamma_star, scenarios_added, solve_stats)
    println("\n====================================================")
    println("FINAL SUMMARY")
    println("  gamma_star           = ", gamma_star)
    println("  scenarios_added      = ", scenarios_added)
    println("  NLPs solved          = ", solve_stats[:nlp_count])
    println("  NLP solve time (s)   = ", round(solve_stats[:nlp_time]; digits=3))
    println("====================================================")
end

