from stein import *
import numpy as np
import cdt

def generate(d, s0, N, noise_std = 1, noise_type = 'Gauss', graph_type = 'ER', GP = True, lengthscale=1):
    adjacency = simulate_dag(d, s0, graph_type, triu=True)
    teacher = Dist(d, noise_std, noise_type, adjacency, GP = GP, lengthscale=lengthscale)
    X, noise_var = teacher.sample(N)
    return X, adjacency


# Data generation paramters
graph_type = 'ER'
d = 10
s0 = 10
N = 1000

X, adj = generate(d, s0, N, GP=True)

# SCORE hyper-parameters
eta_G = 0.001
eta_H = 0.001
cam_cutoff = 0.001

A_SCORE, top_order_SCORE =  SCORE(X, eta_G, eta_H, cam_cutoff)
print("SHD : {}".format(SHD(A_SCORE, adj)))
print("SID: {}".format(int(cdt.metrics.SID(target=adj, pred=A_SCORE))))
print("top order errors: {}".format(num_errors(top_order_SCORE, adj)))
