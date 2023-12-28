import random
from copy import deepcopy
from src.player_class import Player


class PopulationState:
    def __init__(self, Z, poor_proportion):
        self.population_list = [[], []]
        for i in range(Z):
            if i < Z * poor_proportion:
                self.population_list[1].append(Player("poor"))
            else:
                self.population_list[0].append(Player("rich"))

        self.population = [[0, 0], [0, 0]]
        for player in self.population_list[0]:
            if player.strategy == "cooperator":
                self.population[0][0] += 1
            else:
                self.population[0][1] += 1
        for player in self.population_list[1]:
            if player.strategy == "cooperator":
                self.population[0][0] += 1
            else:
                self.population[0][1] += 1

    def change_strategy(self, player):
        if player.strategy == "cooperator":
            player.strategy = "defector"
            if player.wealth == "rich":
                self.population[0][0] -= 1
                self.population[0][1] += 1
            else:
                self.population[1][0] -= 1
                self.population[1][1] += 1
        else:
            player.strategy = "cooperator"
            if player.wealth == "rich":
                self.population[0][0] += 1
                self.population[0][1] -= 1
            else:
                self.population[1][0] += 1
                self.population[1][1] -= 1

    def select(self, h):
        if h == 1:
            p1 = random.choice(self.population_list[0] + self.population_list[1])
            if p1.wealth == "rich":
                list = deepcopy(self.population_list[0])
                if p1 in list:
                    list.remove(p1)
                p2 = random.choice(self.population_list[0])
            else:
                list = deepcopy(self.population_list[1])
                if p1 in list:
                    list.remove(p1)
                p2 = random.choice(self.population_list[1])
        elif h:
            p1 = random.choice(self.population_list[0] + self.population_list[1])
            if p1.wealth == "rich":
                temp = deepcopy(self.population_list[1])
                random.shuffle(temp)
                temp = temp[0:int((1-h)*len(temp))]
                list = self.population_list[0] + temp
            else:
                temp = deepcopy(self.population_list[0])
                random.shuffle(temp)
                temp = temp[0:int((1-h)*len(temp))]
                list = self.population_list[1] + temp
            if p1 in list:
                list.remove(p1)
            p2 = random.choice(list)
        else:
            p1, p2 = random.sample(self.population_list[0] + self.population_list[1], k=2)
        return p1, p2

