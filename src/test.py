from math import exp

import numpy as np
import sys

from matplotlib import pyplot as plt

import pgg_game

np.set_printoptions(threshold=sys.maxsize)

# i = [[115, 35], [35, 15]]
Z = 100
Zr = 20
Zp = 80
Z_tot = [Zr, Zp]
N = 6
mu = 1/Z
h = 1
beta = 0.1

b_r = b_p = 0.1
c_r = c_p = 0.1
Mcb_threshold = 6
r = 0.2

T_coop_rich = 1
T_defect_rich = 1
T_coop_poor = 1

nb_strat = 2


def fermi_fun(a, b, i, beta):
    if a[0] == 0:
        wealth_class_A = "rich"
    else:
        wealth_class_A = "poor"

    if a[1] == 0:
        strat_A = "cooperator"
    else:
        strat_A = "defector"

    if b[0] == 0:
        wealth_class_B = "rich"
    else:
        wealth_class_B = "poor"

    if b[1] == 0:
        strat_B = "cooperator"
    else:
        strat_B = "defector"

    res = 0

    if not i[b[0]][b[1]] == 0:
        res = 1 / (1 + exp(beta * (
            pgg_game.fitness(wealth_class_A, strat_A, i, Z, N, b_r, b_p, c_r, c_p,
                             Mcb_threshold, r)
            - pgg_game.fitness(wealth_class_B, strat_B, i, Z, N, b_r, b_p, c_r, c_p,
                               Mcb_threshold, r))))

    return res


def compute_T(i, k, l, X, Y, Z, mu, beta, Z_tot, h):
    T_k_XtoY = (i[k][X] / Z) * (mu + (1 - mu) *
                                (i[k][Y] / (Z_tot[k] - 1 + (1 - h) * Z_tot[l]) * (
                                        fermi_fun([k, X], [k, Y], i, beta)) +
                                 (1 - h) * i[l][Y] / (Z_tot[k] - 1 + (1 - h) * Z_tot[l]) * (
                                         fermi_fun([k, X], [l, Y], i, beta)))
                                )
    return T_k_XtoY


M = np.zeros([(Zp + 1) * (Zr + 1), (Zp + 1) * (Zr + 1)])
a_g = []
for rich_coop in range(Zr + 1):
    for poor_coop in range(Zp + 1):
        i = [[rich_coop, Zr - rich_coop], [poor_coop, Zp - poor_coop]]

        for k in range(nb_strat):
            l = (k + 1) % nb_strat
            for v in range(nb_strat):
                X = v
                Y = (v + 1) % nb_strat
                if k == 0:
                    if X == 0:
                        if rich_coop != 0:
                            T_k_XtoY = compute_T(i, k, l, X, Y, Z, mu, beta, Z_tot, h)
                            M[rich_coop * (Zp+1) + poor_coop][(rich_coop - 1) * (Zp+1) + poor_coop] = T_k_XtoY
                    elif X == 1:
                        if rich_coop != Zr:
                            T_k_XtoY = compute_T(i, k, l, X, Y, Z, mu, beta, Z_tot, h)
                            M[rich_coop * (Zp+1) + poor_coop][(rich_coop + 1) * (Zp+1) + poor_coop] = T_k_XtoY
                else:
                    if X == 0:
                        if poor_coop != 0:
                            T_k_XtoY = compute_T(i, k, l, X, Y, Z, mu, beta, Z_tot, h)
                            M[rich_coop * (Zp+1) + poor_coop][rich_coop * (Zp+1) + (poor_coop - 1)] = T_k_XtoY
                    elif X == 1:
                        if poor_coop != Zp:
                            T_k_XtoY = compute_T(i, k, l, X, Y, Z, mu, beta, Z_tot, h)
                            M[rich_coop * (Zp+1) + poor_coop][rich_coop * (Zp+1) + (poor_coop + 1)] = T_k_XtoY

        M[rich_coop * (Zp+1) + poor_coop][rich_coop * (Zp+1) + poor_coop] = 1 - np.sum(M[rich_coop * (Zp+1) + poor_coop])

        if poor_coop != 0 and rich_coop != 0 and rich_coop != Zr and poor_coop != Zp:
            a_g.append(pgg_game.fitness("rich", "cooperator", i, Z, N, b_r, b_p, c_r, c_p,Mcb_threshold, r))

 # pgg_game.fitness("rich", "defector", i, Z, N, b_r, b_p, c_r, c_p,Mcb_threshold, r) +
 # pgg_game.fitness("poor", "cooperator", i, Z, N, b_r, b_p, c_r, c_p,Mcb_threshold, r) +
 # pgg_game.fitness("poor", "defector", i, Z, N, b_r, b_p, c_r, c_p,Mcb_threshold, r)

w, v = np.linalg.eig(M.transpose())

j_stationary = np.argmin(abs(w-1.0))

p_stationary=abs(v[:, j_stationary].real)

p_norm = p_stationary / p_stationary.sum()

xs = [x for x in range(len(p_stationary))]

plt.plot(xs, p_norm)
plt.show()
