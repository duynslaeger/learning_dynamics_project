import math
import random
import numpy as np
import pgg_game

'''
- group_size (int)    = 6        : Size of a group 
- b_r (float)         = 2.5      : initial endowment for rich individuals 
- b_p (float)         = 0.625    : initial endowment for poor individuals
- c_r (float)         = 0.1*b_r  : the proportion of endowment given by rich cooperator (equals to c*b_r)
- c_p (float)         = 0.1*b_p  : the proportion of endowment given by poor cooperator (equals to c*b_p)
- Mcb_threshold (int) = 3*0.1*1  : threshold that the group needs to reach if they want to win. Equals to M*c*b_av (b_av is the average endowment)
- r (float)           = 0.3      : perception of risk
'''

def prob_imitation(beta, fitness):
    p = 1/(1 + math.exp(beta * (fitness[0] - fitness[1])))
    return p


def moran_step(population_state, beta, mu, Z, h):
    p1, p2 = population_state.select(h)
    fitness = pgg_game.fitness(p1.wealth, p1.strategy, population_state.population, Z, 6, 2.5, 0.625, 0.25, 0.0625, 0.3, 0.3), pgg_game.fitness(p2.wealth, p2.strategy, population_state.population, Z, 6, 2.5, 0.625, 0.25, 0.0625, 0.3, 0.3)
    if np.random.rand() < mu:
        if random.randint(0,1) == 1:
            population_state.change_strategy(p1)
    elif np.random.rand() < prob_imitation(beta, fitness):
        if p1.strategy != p2.strategy:
            population_state.change_strategy(p1)


def estimate_stationary_distribution(nb_runs, transitory, nb_generations, beta, mu, Z, h, population_state):
	count_list = [0]*Z
	for i in range(nb_runs):
		for i in range(transitory):
			moran_step(population_state, beta, mu, Z, h)
		for i in range(nb_generations-transitory):
			moran_step(population_state, beta, mu, Z, h)
			count_list[population_state.count_cooperate_player()-1]+=1
	for i in range(len(count_list)):
		count_list[i]/=(nb_generations-transitory)
	return count_list
	
