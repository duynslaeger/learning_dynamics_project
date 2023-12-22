import numpy as np


def heaviside(k):
    if(k >= 0):
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

    
    if(wealth_class == "poor"):
        b_rp = b_p
    elif(wealth_class == "rich"):
        b_rp = b_r
    else:
        raise ValueError('Wrong argument given in the wealth_class argument of the personnal_payoff_defector() function. Shoulb be "rich" or "poor".')
    
    result = b_rp * (heaviside(c_r*j_r + c_p*j_p - Mcb_threshold) + (1-r)*(1 - heaviside(c_r*j_r + c_p*j_p - Mcb_threshold)))
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

    
    if(wealth_class == "poor"):
        c_rp = c_p
    elif(wealth_class == "rich"):
        c_rp = c_r
    else:
        raise ValueError('Wrong argument given in the wealth_class argument of the personnal_payoff_defector() function. Shoulb be "rich" or "poor".')
    
    result = personnal_payoff_defector(wealth_class, j_r, j_p, b_r, b_p, c_r, c_p, Mcb_threshold, r) - c_rp
    return result