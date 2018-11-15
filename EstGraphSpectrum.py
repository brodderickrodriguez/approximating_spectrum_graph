
import numpy as np
import random as rd
import recover_density


def est_graph_spectrum(edge_list, degree, start, s, k, x):
    n = len(degree[0])
    ag_pdf = np.zeros(len(x))
    trials = 20

    for trial in range(trials):
        h = np.zeros(k)
        x_count = np.zeros(1, k)

        for i in range(s):
            j = rd.randint(n)
            w = j

            for step in range(k):
                index = start[w] + rd.randint(degree[w] - 1)
                w = edge_list[index][1]

                if w == j:
                    x_count[step] = x_count[step] + 1

        for i in range(k):
            h[i] = x_count[i] / s

        f_k = np.ones(1, k)
        (rec_pdf, t) = recover_density.recover_density(h, x, f_k)

        ag_pdf += rec_pdf

    ag_pdf /= trials
    return ag_pdf
