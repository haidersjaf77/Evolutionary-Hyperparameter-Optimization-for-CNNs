def select_parents(population, fitness, num_parents):
    parents = np.array(population)[np.argsort([f[0] for f in fitness])[-num_parents:]].tolist()
    return parents