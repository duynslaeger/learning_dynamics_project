from math import exp
from scipy.stats import multivariate_hypergeom
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

b_r = 2.5
b_p = 0.625
c_r = 0.1 * b_r
c_p = 0.1 * b_p

b_bar = 1
c_bar = 0.1

M_thresh = 3

Mcb_threshold = c_bar * M_thresh * b_bar
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


def compute_T(i, k, l, X, Y, Z, mu, beta, Z_tot, h, obstinators):
    # T_k_XtoY = (i[k][X] / Z) * ((i[k][X] - obstinators) / i[k][X]) * (mu + (1 - mu) *
    #                             (i[k][Y] / (Z_tot[k] - 1 + (1 - h) * Z_tot[l]) * (
    #                                 fermi_fun([k, X], [k, Y], i, beta)) +
    #                              (1 - h) * i[l][Y] / (Z_tot[k] - 1 + (1 - h) * Z_tot[l]) * (
    #                                  fermi_fun([k, X], [l, Y], i, beta)))
    #                             )
    T_k_XtoY = ((i[k][X] - obstinators) / Z) * (mu + (1 - mu) *
                                (i[k][Y] / (Z_tot[k] - 1 + (1 - h) * Z_tot[l]) * (
                                    fermi_fun([k, X], [k, Y], i, beta)) +
                                 (1 - h) * i[l][Y] / (Z_tot[k] - 1 + (1 - h) * Z_tot[l]) * (
                                     fermi_fun([k, X], [l, Y], i, beta)))
                                )
    return T_k_XtoY

eta_g_over_risk = []
x_axis = [0.2]
for ind_x in range(len(x_axis)):
    r = x_axis[ind_x]
    print("Percieved risk = ", r)
    M = np.zeros([(Zp + 1) * (Zr + 1), (Zp + 1) * (Zr + 1)])
    a_g = []
    M_plot = np.zeros([Zp + 1, Zr + 1])
    M1 = [[[0,0,0,0] for _ in range(Zp + 1)] for _ in range(Zr + 1)]
    for rich_coop in tqdm(range(obstinate_coop[0], Zr + 1 - obstinate_defector[0])):
        for poor_coop in range(obstinate_coop[1], Zp + 1 - obstinate_defector[1]):
            i = [[rich_coop, Zr - rich_coop], [poor_coop, Zp - poor_coop]]
            for k in range(nb_strat):
                l = (k + 1) % nb_strat
                for v in range(nb_strat):
                    X = v
                    Y = (v + 1) % nb_strat
                    if k == 0:
                        if X == 0:
                            if rich_coop != 0:
                                T_k_XtoY = compute_T(i, k, l, X, Y, Z, mu, beta, Z_tot, h, obstinate_coop[0])
                                M[rich_coop * (Zp + 1) + poor_coop][(rich_coop - 1) * (Zp + 1) + poor_coop] = T_k_XtoY
                                M1[rich_coop][poor_coop][0] = T_k_XtoY
                        elif X == 1:
                            if rich_coop != Zr:
                                T_k_XtoY = compute_T(i, k, l, X, Y, Z, mu, beta, Z_tot, h, obstinate_defector[0])
                                M[rich_coop * (Zp + 1) + poor_coop][(rich_coop + 1) * (Zp + 1) + poor_coop] = T_k_XtoY
                                M1[rich_coop][poor_coop][1] = T_k_XtoY
                    else:
                        if X == 0:
                            if poor_coop != 0:
                                T_k_XtoY = compute_T(i, k, l, X, Y, Z, mu, beta, Z_tot, h, obstinate_coop[1])
                                M[rich_coop * (Zp + 1) + poor_coop][rich_coop * (Zp + 1) + (poor_coop - 1)] = T_k_XtoY
                                M1[rich_coop][poor_coop][2] = T_k_XtoY
                        elif X == 1:
                            if poor_coop != Zp:
                                T_k_XtoY = compute_T(i, k, l, X, Y, Z, mu, beta, Z_tot, h, obstinate_defector[1])
                                M[rich_coop * (Zp + 1) + poor_coop][rich_coop * (Zp + 1) + (poor_coop + 1)] = T_k_XtoY
                                M1[rich_coop][poor_coop][3] = T_k_XtoY

            M[rich_coop * (Zp + 1) + poor_coop][rich_coop * (Zp + 1) + poor_coop] = 1 - np.sum(M[rich_coop * (Zp + 1) + poor_coop])

            rich_defectors = Zr - rich_coop
            poor_defectors = Zp - poor_coop
            multi_hypergeom_curr_composition = multivariate_hypergeom(m = [rich_coop, rich_defectors, poor_coop, poor_defectors], n = 6)
            aG_i = 0.0
            for rc in range(N): # rc represent the number of rich cooperator in the group
                for pc in range(N) : # pc represent the number of poor cooperator in the group
                    if(rc + pc) > M_thresh and(rc + pc) <= N: # if there are enough cooperators to reach the threshold
                        for rd in range(N): # rd represent the number of rich defectors in the group
                            if(rc + pc + rd <= N):
                                for pd in range(N): # pd represent the number of poor defectors in the group
                                    if(rc + pc + rd + pd == N):
                                        pmf = multi_hypergeom_curr_composition.pmf(x=[rc, rd, pc, pd])
                                        aG_i += pmf
            a_g.append(aG_i)

    Y, X = np.mgrid[0:Zp:(Zp+1)*1j, 0:Zr:(Zr+1)*1j]
    U = X  # M1[X][Y][2] - M1[X][Y][3]
    V = Y  # M1[X][Y][1] - M1[X][Y][0]

    U = np.array([[(M1[i][j][1] - M1[i][j][0]) for i in range(Zr + 1)] for j in range(Zp + 1)])
    V = np.array([[(M1[i][j][3] - M1[i][j][2]) for i in range(Zr + 1)] for j in range(Zp + 1)])
    speed = np.sqrt(U**2 + V**2)
    lw = 5 * speed / speed.max()

    plt.streamplot(X, Y, U, V, density=0.8, color=speed, linewidth=2, cmap='viridis')
    plt.colorbar(label='Speed')  # Add colorbar to show speed values

    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title('Vector Field with Transition Matrix')
    plt.grid()
    plt.show()

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
    eta_g_over_risk.append(eta_g)


# xs = [x for x in range(len(p_stationary))]
# plt.plot(xs, p_norm)
# plt.show()

# plt.matshow(M_plot, cmap=plt.get_cmap('gray'), vmin=0.0, vmax=1.0)
# plt.show()


plt.plot(x_axis,eta_g_over_risk, color='r', label="h = 1")
plt.ylim([0.0, 1.0])
plt.xlabel("risk (r)")
plt.ylabel("Group achievement ($\\eta_G$)")
plt.legend()
plt.show()
