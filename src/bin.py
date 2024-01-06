from scipy.stats import multivariate_hypergeom

Zr = 20
Zp = 80
M_thresh = 3
N = 6

rich_coop = 20
poor_coop = 80
multi_hypergeom_curr_composition = multivariate_hypergeom(m = [10, 10, 40, 40], n = 6)

rich_defectors = Zr - rich_coop
poor_defectors = Zp - poor_coop
# [2, 2, 1, 1]

aG_i = 0.0
count = 0

# for rc in range(N): # rc represent the number of rich cooperator in the group
#     for pc in range(N - rc) : # pc represent the number of poor cooperator in the group
#         if(rc + pc) > M_thresh: # if there are enough cooperators to reach the threshold
#             for rd in range(N-rc-pc, -1, -1): # rd represent the number of rich defectors in the group
#                 for pd in range(N - rc - pc - rd, -1, -1): # pd represent the number of poor defectors in the group
#                     print("here")
#                     print(rc, rd, pc, pd)
#                     pmf = multi_hypergeom_curr_composition.pmf(x=[rc, rd, pc, pd])
#                     print(pmf)
#                     aG_i += pmf


for rc in range(N): # rc represent the number of rich cooperator in the group
    for pc in range(N) : # pc represent the number of poor cooperator in the group
        if(rc + pc) > M_thresh and(rc + pc) <= N: # if there are enough cooperators to reach the threshold
            for rd in range(N): # rd represent the number of rich defectors in the group
                if(rc + pc + rd <= N):
                    for pd in range(N): # pd represent the number of poor defectors in the group
                        if(rc + pc + rd + pd == N):
                            pmf = multi_hypergeom_curr_composition.pmf(x=[rc, rd, pc, pd])
                            # count += 1
                            aG_i += pmf


print(count)
print(aG_i)