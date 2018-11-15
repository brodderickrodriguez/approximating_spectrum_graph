import numpy as np
import scipy as sp
def recover_density(H, x, f_k):
    k = H.shape[1]
    ns = x.shape[1]
    V = np.ones((3,5))
    V[0,] = x

    for i in range(1, k+1):
        V[i,] = np.multiply(V(i-1,), x)
    
    neg_eye_k = np.negative(np.identity(k))
    zeros_k_1 = np.zeros((k,1))
    zeros_1_k = np.zeros((1,k)) 
    ones_1_ns = np.ones((1, ns))
    A = np.vstack(
        np.hstack(V, neg_eye_k, zeros_k_1),
        np.hstack(np.negative(V), neg_eye_k, zeros_k_1),
        np.hstack(ones_1_ns, zeros_1_k, -1),
        np.hstack(np.negative(ones_1_ns), zeros_1_k, -1)
    )

    b = np.hstack(H.conj().transpose(), np.negative(H).conj().transpose, 1, -1)
    f = np.hstack(np.zeros((1, ns)), f_k, 1)
    lb = np.zeros((ns+k+1, 1))
    Nnorm = np.divide(np.identity(ns), 10)
    opt_res = sp.optimize.linprog(f, A, b)
    pdf = opt_res.x[0:ns].conj().transpose()
    t = np.subtract(b[1:k], np.multiply(opt_res.x[1:ns]))

    return (pdf, t)