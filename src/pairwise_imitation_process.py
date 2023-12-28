import math
import random
import numpy as np
import pgg_game


def prob_imitation(beta, fitness):
    p = 1/(1 + math.exp(beta * (fitness[0] - fitness[1])))
    return p


def moran_step(population_state, beta, mu, Z, h):
    p1, p2 = population_state.select(h)
    fitness = pgg_game.fitness(p1.wealth, p1.strategy, population_state.population, Z), pgg_game.fitness(p2.wealth, p2.strategy, population_state.population, Z)
    if np.random.rand() < mu:
        if random.randint(0,1) == 1:
            population_state.change_strategy(p1)
    elif np.random.rand() < prob_imitation(beta, fitness):
        if p1.strategy != p2.strategy:
            population_state.change_strategy(p1)


def estimate_stationary_distribution(nb_runs, transitory, nb_generations, beta, mu, Z, h, population_state):
    for i in range(nb_runs):
        for i in range(transitory):
            moran_step(population_state, beta, mu, Z, h)
        for i in range(nb_generations-transitory):
            moran_step(population_state, beta, mu, Z, h)
