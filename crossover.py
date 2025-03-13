def crossover(parent1, parent2):
    child = {}
    for key in parent1.keys():
        child[key] = parent1[key] if np.random.rand() < 0.5 else parent2[key]
    return child