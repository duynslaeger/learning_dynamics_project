import random


class Player:
    def __init__(self, wealth):
        self.wealth = wealth
        self.strategy = random.choice(["cooperator", "defector"])
