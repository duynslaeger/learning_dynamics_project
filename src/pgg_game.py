import numpy as np
from math import comb

# population = [[R_C, R_P],[P_C, P_P]]

def heaviside(k):
    if (k >= 0):
        return 1
    else:
        return 0


def personnal_payoff_defector(wealth_class, j_r, j_p, b_r, b_p, c_r, c_p, Mcb_threshold, r):
    """Returns the payoff of a defector inside a group of j_r rich 
        individuals and j_p poor individuals

    Parameters
    ----------
    - wealth_class (str)  : Represents the wealth class of the individual. Should be either "rich" or "poor".
    - j_r (int)           : number of rich individuals inside the group
    - j_p (int)           : number of poor individuals inside the group
    - b_r (float)         : initial endowment for rich individuals
    - b_p (float)         : initial endowment for poor individuals
    - c_r (float)         : the proportion of endowment given by rich cooperator (equals to c*b_r)
    - c_p (float)         : the proportion of endowment given by poor cooperator (equals to c*b_p)
    - Mcb_threshold (int) : threshold that the group needs to reach if they want to win. Equals to M*c*b_av (b_av is the average endowment)
    - r (float)           : perception of risk

    Returns
    -------
    The personnal payoff of a defector individual (float)
    """

    if (wealth_class == "poor"):
        b_rp = b_p
    elif (wealth_class == "rich"):
        b_rp = b_r
    else:
        raise ValueError(
            'Wrong argument given in the wealth_class argument of the personnal_payoff_defector() function. Shoulb be "rich" or "poor".')

    result = b_rp * (heaviside(c_r * j_r + c_p * j_p - Mcb_threshold) + (1 - r) * (
            1 - heaviside(c_r * j_r + c_p * j_p - Mcb_threshold)))
    return result


def personnal_payoff_cooperator(wealth_class, j_r, j_p, b_r, b_p, c_r, c_p, Mcb_threshold, r):
    """Returns the payoff of a cooperator inside a group of j_r rich 
        individuals and j_p poor individuals

    Parameters
    ----------
    - wealth_class (str)  : Represents the wealth class of the individual. Should be either "rich" or "poor".
    - j_r (int)           : number of rich individuals inside the group
    - j_p (int)           : number of poor individuals inside the group
    - b_r (float)         : initial endowment for rich individuals
    - b_p (float)         : initial endowment for poor individuals
    - c_r (float)         : the proportion of endowment given by rich cooperator (equals to c*b_r)
    - c_p (float)         : the proportion of endowment given by poor cooperator (equals to c*b_p)
    - Mcb_threshold (int) : threshold that the group needs to reach if they want to win. Equals to M*c*b_av (b_av is the average endowment)
    - r (float)           : perception of risk

    Returns
    -------
    The personnal payoff of a cooperator individual (float)
    """

    if (wealth_class == "poor"):
        c_rp = c_p
    elif (wealth_class == "rich"):
        c_rp = c_r
    else:
        raise ValueError(
            'Wrong argument given in the wealth_class argument of the personnal_payoff_defector() function. Shoulb be "rich" or "poor".')

    result = personnal_payoff_defector(wealth_class, j_r, j_p, b_r, b_p, c_r, c_p, Mcb_threshold, r) - c_rp
    return result

def pass_threshold(j_r, j_p, c_r, c_p, Mcb_threshold):
    return heaviside(c_r * j_r + c_p * j_p - Mcb_threshold)


def fitness(wealth_class, strategy, population, population_size, group_size, b_r, b_p, c_r, c_p, Mcb_threshold, r):
    """Returns the fitness of a specific kind of individual with a specific strategy in the population

        Parameters
        ----------
        - wealth_class (str)   : Represents the wealth class of the individual. Should be either "rich" or "poor".
        - strategy (str)       : Represent the strategy which you want to compute the fitness
        - population (list)    : Represent the state of the population
        - population_size (int): Size of the population
        - group_size (int)     : Size of a group
        - b_r (float)          : initial endowment for rich individuals
        - b_p (float)          : initial endowment for poor individuals
        - c_r (float)          : the proportion of endowment given by rich cooperator (equals to c*b_r)
        - c_p (float)          : the proportion of endowment given by poor cooperator (equals to c*b_p)
        - Mcb_threshold (int)  : threshold that the group needs to reach if they want to win. Equals to M*c*b_av (b_av is the average endowment)
        - r (float)            : perception of risk

        Returns
        -------
        The fitness of an individual with a specific strategy in a given population state (float)
        """
    result = 0
    pop_coef = 1 / comb(population_size - 1, group_size - 1)
    i_rich, i_poor = population[0][0], population[1][0] #[sublist[0] + sublist[1] for sublist in population]

    if wealth_class == "poor":
        if strategy == "cooperator":
            #i_rich, i_poor = [sublist[0] for sublist in population]
            for j_rich in range(group_size):
                for j_poor in range(group_size - j_rich):
                    rich_coef = comb(i_rich, j_rich)
                    poor_coef = comb(i_poor - 1, j_poor)
                    pop_group_coef = comb(population_size - i_rich - i_poor, group_size - 1 - j_rich - j_poor)
                    pay_off = personnal_payoff_cooperator(wealth_class, j_rich, j_poor+1, b_r, b_p, c_r, c_p,
                                                          Mcb_threshold, r)
                    result += rich_coef * poor_coef * pop_group_coef * pay_off
            result *= pop_coef

        if strategy == "defector":
            #i_rich, i_poor = [sublist[1] for sublist in population]
            for j_rich in range(group_size):
                for j_poor in range(group_size - j_rich):
                    rich_coef = comb(i_rich, j_rich)
                    poor_coef = comb(i_poor, j_poor)
                    pop_group_coef = comb(population_size -1 - i_rich - i_poor, group_size - 1 - j_rich - j_poor)
                    pay_off = personnal_payoff_defector(wealth_class, j_rich, j_poor, b_r, b_p, c_r, c_p,
                                                          Mcb_threshold, r)
                    result += rich_coef * poor_coef * pop_group_coef * pay_off
            result *= pop_coef

    elif wealth_class == "rich":
        if strategy == "cooperator":
            #i_rich, i_poor = [sublist[0] for sublist in population]
            for j_rich in range(group_size):
                for j_poor in range(group_size-j_rich):
                    rich_coef = comb(abs(i_rich-1), j_rich)
                    poor_coef = comb(i_poor, j_poor)
                    pop_group_coef = comb(population_size-i_rich-i_poor, group_size-1-j_rich-j_poor)
                    pay_off = personnal_payoff_cooperator(wealth_class, j_rich+1, j_poor, b_r, b_p, c_r, c_p,
                                                          Mcb_threshold, r)
                    result += rich_coef*poor_coef*pop_group_coef*pay_off
            result *= pop_coef

        if strategy == "defector":
            #i_rich, i_poor = [sublist[1] for sublist in population]
            for j_rich in range(group_size):
                for j_poor in range(group_size - j_rich):
                    rich_coef = comb(i_rich, j_rich)
                    poor_coef = comb(i_poor, j_poor)
                    pop_group_coef = comb(population_size - 1 - i_rich - i_poor, group_size - 1 - j_rich - j_poor)
                    pay_off = personnal_payoff_defector(wealth_class, j_rich, j_poor, b_r, b_p, c_r, c_p,
                                                          Mcb_threshold, r)
                    result += rich_coef * poor_coef * pop_group_coef * pay_off
            result *= pop_coef
    else:
        raise ValueError(
            'Wrong argument given in the wealth_class argument of the fitness() function. Shoulb be "rich" or "poor".')

    return result

def fraction_group(wealth_class, strategy, population, population_size, group_size, b_r, b_p, c_r, c_p, Mcb_threshold, r):
    """Returns the fitness of a specific kind of individual with a specific strategy in the population

        Parameters
        ----------
        - wealth_class (str)   : Represents the wealth class of the individual. Should be either "rich" or "poor".
        - strategy (str)       : Represent the strategy which you want to compute the fitness
        - population (list)    : Represent the state of the population
        - population_size (int): Size of the population
        - group_size (int)     : Size of a group
        - b_r (float)          : initial endowment for rich individuals
        - b_p (float)          : initial endowment for poor individuals
        - c_r (float)          : the proportion of endowment given by rich cooperator (equals to c*b_r)
        - c_p (float)          : the proportion of endowment given by poor cooperator (equals to c*b_p)
        - Mcb_threshold (int)  : threshold that the group needs to reach if they want to win. Equals to M*c*b_av (b_av is the average endowment)
        - r (float)            : perception of risk

        Returns
        -------
        The fitness of an individual with a specific strategy in a given population state (float)
        """
    result = 0
    result2 = 0
    pop_coef = 1 / comb(population_size - 1, group_size - 1)
    i_rich, i_poor = population[0][0], population[1][0] #[sublist[0] + sublist[1] for sublist in population]

    if wealth_class == "poor":
        if strategy == "cooperator":
            #i_rich, i_poor = [sublist[0] for sublist in population]
            for j_rich in range(group_size):
                for j_poor in range(group_size - j_rich):
                    rich_coef = comb(i_rich, j_rich)
                    poor_coef = comb(i_poor, j_poor)
                    pop_group_coef = comb(population_size - i_rich - i_poor, group_size - j_rich - j_poor)
                    pay_off = pass_threshold(j_rich, j_poor, c_r, c_p, Mcb_threshold)
                    result += rich_coef * poor_coef * pop_group_coef * pay_off
                    result2 += rich_coef * poor_coef * pop_group_coef
            if result != 0:
                result /= result2

        if strategy == "defector":
            #i_rich, i_poor = [sublist[1] for sublist in population]
            for j_rich in range(group_size):
                for j_poor in range(group_size - j_rich):
                    rich_coef = comb(i_rich, j_rich)
                    poor_coef = comb(i_poor, j_poor)
                    pop_group_coef = comb(population_size -1 - i_rich - i_poor, group_size - 1 - j_rich - j_poor)
                    pay_off = pass_threshold(j_rich, j_poor, b_r, b_p, Mcb_threshold)

                    result += rich_coef * poor_coef * pop_group_coef * pay_off
            result *= pop_coef

    elif wealth_class == "rich":
        if strategy == "cooperator":
            #i_rich, i_poor = [sublist[0] for sublist in population]
            for j_rich in range(group_size):
                for j_poor in range(group_size-j_rich):
                    rich_coef = comb(abs(i_rich-1), j_rich)
                    poor_coef = comb(i_poor, j_poor)
                    pop_group_coef = comb(population_size-i_rich-i_poor, group_size-1-j_rich-j_poor)
                    pay_off = pass_threshold(j_rich, j_poor, b_r, b_p, Mcb_threshold)
                    result += rich_coef*poor_coef*pop_group_coef*pay_off
            result *= pop_coef

        if strategy == "defector":
            #i_rich, i_poor = [sublist[1] for sublist in population]
            for j_rich in range(group_size):
                for j_poor in range(group_size - j_rich):
                    rich_coef = comb(i_rich, j_rich)
                    poor_coef = comb(i_poor, j_poor)
                    pop_group_coef = comb(population_size - 1 - i_rich - i_poor, group_size - 1 - j_rich - j_poor)
                    pay_off = pass_threshold(j_rich, j_poor, b_r, b_p, Mcb_threshold)
                    result += rich_coef * poor_coef * pop_group_coef * pay_off
            result *= pop_coef
    else:
        raise ValueError(
            'Wrong argument given in the wealth_class argument of the fitness() function. Shoulb be "rich" or "poor".')

    return result

