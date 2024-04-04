def should_stop_convergence(change_in_objective, objective_threshold=1e-6):
    if abs(change_in_objective) < objective_threshold:
        return True
    return False
