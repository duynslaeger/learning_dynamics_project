import pairwise_imitation_process
from src.populationState_class import PopulationState

Z = 200
poor_proportion = 0.8
beta = 10
mu = 1/Z
transitory = 1000
nb_generations = 10000
nb_runs = 10
homophily = 0
population_state = PopulationState(Z, poor_proportion)

pairwise_imitation_process.estimate_stationary_distribution(nb_runs, transitory, nb_generations, beta, mu, Z, homophily,
                                                            population_state)
