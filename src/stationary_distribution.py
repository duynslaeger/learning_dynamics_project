import pairwise_imitation_process
import random
import math
import matplotlib.pyplot as plt
from populationState_class import PopulationState

def main():
    """ Returns the stationary distribution	"""
	Z = 200
	poor_proportion = 0.8
	beta = 10
	mu = 1/Z
	transitory = 100
	nb_generations = 1000
	nb_runs = 5
	homophily = 0
	population_state = PopulationState(Z, poor_proportion)
	vector = pairwise_imitation_process.estimate_stationary_distribution(nb_runs, transitory, nb_generations, beta, mu, Z, homophily, population_state)
	fix, ax = plt.subplots(figsize=(8, 5))
	ax.set_ylabel('stationary distribution', fontsize=15, fontweight='bold')
	ax.set_xlabel('frequency of cooperator (k/Z)', fontsize=15, fontweight='bold')
	plt.plot([i/Z for i in range(1, len(vector) + 1)],vector)
	plt.savefig('../result/moran_process.png')
if __name__ == "__main__":
	main()
