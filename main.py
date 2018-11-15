import numpy as np
import EstGraphSpectrum as egs
import pdf2cdf as p2c
import matplotlib.pyplot as plt

#edge_list = np.array([[0, 1], [3, 2], [0, 3]])
p = 0.1
n = 500
graph = np.triu((1, p, 500, 500), 1)
graph = graph + np.conj().transpose()
deg = np.zeros(n, 1)
start = np.zeros(n, 1)
start[0] = 1
edge_list = []

for i in range(n):
    e_list = np.nonzero(graph[i, ]).conj().transpose()

    deg[i] = len(edge_list[0])
    start[i + 1] = start[i] + deg[i]
    #e_list = [i * np.ones(deg[i], 0), e_list]
    #edge_list = [edge_list, e_list]

start = start[1:n]
n_grid = 1001
x = np.zeros(1, n_grid)

for t in range(n_grid):
    x[t] = np.cos((2 * (t - 1)) / (2 * n_grid * np.pi))

x = np.flip(x)
s = 1000
k = 20
rec_pdf = egs.est_graph_spectrum(edge_list, deg, start, s, k, x)
rec_pdf = p2c.pdf_to_cdf(rec_pdf)

# TODO: plot this
'''
figure
hold on
scatter(rec_cdf,1-x);
% compute groundtrurth
L = eye(n)-diag(deg)^(-1/2)*G*diag(deg)^(-1/2);
scatter(1/n:1/n:1,svd(L));
legend('ground-truth','recovery');
hold off
'''
