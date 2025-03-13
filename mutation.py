def mutate(individual):
    if np.random.rand() < 0.2:
        individual['learning_rate'] *= np.random.choice([0.5, 1.5])
    if np.random.rand() < 0.2:
        individual['batch_size'] = np.random.choice([32, 64, 128])
    if np.random.rand() < 0.2:
        individual['dropout'] = np.random.uniform(0, 0.5)
    return individual