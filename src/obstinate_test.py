from math import exp

import numpy as np
import sys

from matplotlib import pyplot as plt

from tqdm import tqdm

import pgg_game

np.set_printoptions(threshold=sys.maxsize)

# i = [[115, 35], [35, 15]]
Z = 100
Zr = int(Z * 0.2)
Zp = int(Z * 0.8)
Z_tot = [Zr, Zp]
N = 6
mu = 1 / Z
h = 1
obstinate_r_c = 0
obstinate_p_c = 0.1
obstinate_r_d = 0
obstinate_p_d = 0
obstinate_coop = [int(Zr * obstinate_r_c), int(Zp * obstinate_p_c)]
obstinate_defector = [int(Zr * obstinate_r_d), int(Zp * obstinate_p_d)]
beta = 3
Zr -= obstinate_coop[0] + obstinate_defector[0]
Zp -= obstinate_coop[1] + obstinate_defector[1]


b_r = 2.5
b_p = 0.625
c_r = 0.1 * b_r
c_p = 0.1 * b_p

b_bar = 1
c_bar = 0.1

M = 3

Mcb_threshold = c_bar * M * b_bar
r = 0.2

T_coop_rich = 1
T_defect_rich = 1
T_coop_poor = 1

nb_strat = 2


def fermi_fun(a, b, i, beta, Z):
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


def compute_T(i, k, l, X, Y, Z, mu, beta, Z_tot, h, obstinators):
    T_k_XtoY = ((i[k][X] - obstinators) / (Z_tot[0] + Z_tot[1])) * (mu + (1 - mu) *
                                (i[k][Y] / (Z_tot[k] - 1 + (1 - h) * Z_tot[l]) * (
                                    fermi_fun([k, X], [k, Y], i, beta, Z)) +
                                 (1 - h) * i[l][Y] / (Z_tot[k] - 1 + (1 - h) * Z_tot[l]) * (
                                     fermi_fun([k, X], [l, Y], i, beta, Z)))
                                )
    return T_k_XtoY


M = np.zeros([(Zp + 1) * (Zr + 1), (Zp + 1) * (Zr + 1)])
a_g = []
M_plot = np.zeros([Zp + 1, Zr + 1])
for rich_coop in tqdm(range(Zr + 1)):
    for poor_coop in range(Zp + 1):
        i = [[rich_coop + obstinate_coop[0], Zr - rich_coop + obstinate_defector[0]], [poor_coop + obstinate_coop[1], Zp - poor_coop + obstinate_defector[1]]]
        for k in range(nb_strat):
            l = (k + 1) % nb_strat
            for v in range(nb_strat):
                X = v
                Y = (v + 1) % nb_strat
                if k == 0:
                    if X == 0:
                        if rich_coop != 0 + obstinate_coop[0]:
                            T_k_XtoY = compute_T(i, k, l, X, Y, Z, mu, beta, Z_tot, h, obstinate_coop[0])
                            M[rich_coop * (Zp + 1) + poor_coop][(rich_coop - 1) * (Zp + 1) + poor_coop] = T_k_XtoY
                    elif X == 1:
                        if rich_coop != Zr:
                            T_k_XtoY = compute_T(i, k, l, X, Y, Z, mu, beta, Z_tot, h, obstinate_defector[0])
                            M[rich_coop * (Zp + 1) + poor_coop][(rich_coop + 1) * (Zp + 1) + poor_coop] = T_k_XtoY
                else:
                    if X == 0:
                        if poor_coop != 0 + obstinate_coop[1]:
                            T_k_XtoY = compute_T(i, k, l, X, Y, Z, mu, beta, Z_tot, h, obstinate_coop[1])
                            M[rich_coop * (Zp + 1) + poor_coop][rich_coop * (Zp + 1) + (poor_coop - 1)] = T_k_XtoY
                    elif X == 1:
                        if poor_coop != Zp:
                            T_k_XtoY = compute_T(i, k, l, X, Y, Z, mu, beta, Z_tot, h, obstinate_defector[1])
                            M[rich_coop * (Zp + 1) + poor_coop][rich_coop * (Zp + 1) + (poor_coop + 1)] = T_k_XtoY

        M[rich_coop * (Zp + 1) + poor_coop][rich_coop * (Zp + 1) + poor_coop] = 1 - np.sum(
            M[rich_coop * (Zp + 1) + poor_coop])

print("Compute eigen values")
w, v = np.linalg.eig(M.transpose())
print("Compute j_stationary")
j_stationary = np.argmin(abs(w - 1.0))
print("Compute p_stationary")
p_stationary = abs(v[:, j_stationary].real)

p_plot = 1 - (p_stationary / np.max(p_stationary))

p_norm = p_stationary / p_stationary.sum()

for i in range(len(p_plot)):
    M_plot[i % (Zp + 1)][i // (Zp + 1)] = p_plot[i]

xs = [x for x in range(len(p_stationary))]

eta_g = 0
for i in range(len(a_g)):
    eta_g += p_norm[i] * a_g[i]

print(eta_g)

plt.plot(xs, p_norm)
plt.show()

plt.matshow(M_plot, cmap=plt.get_cmap('gray'), vmin=0.0, vmax=1.0)
plt.show()
