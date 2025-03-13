results = {}

def evolutionary_algorithm_with_tracking(generations, population_size, num_parents):
    global results
    population = initialize_population(population_size, X_train.shape[1], y_train.shape[1])

    for generation in range(generations):
        fitness = [evaluate_individual(ind) for ind in population]
        parents = select_parents(population, fitness, num_parents)
        offspring = []

        for _ in range(population_size - num_parents):
            p1, p2 = np.random.choice(parents, size=2, replace=False)
            child = crossover(p1, p2)
            child = mutate(child)
            offspring.append(child)

        population = parents + offspring

    best_individual = population[np.argmax([evaluate_individual(ind)[0] for ind in population])]
    accuracy, precision, recall, f1, training_time, y_test_classes, y_pred_classes = evaluate_individual(best_individual)

    results['Evolutionary Algorithm'] = {
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1,
        'Training Time': training_time,
    }
