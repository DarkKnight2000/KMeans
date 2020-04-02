import numpy as np
import random
from lsalgo import cost_km, Complement
from orc import getClusters
from data import make_data, plotGraph
from main import loss

def kmeans_(U, k, z, eps = 0.0001):
    Uc = [i for i in range(len(U))]
    random.shuffle(Uc)
    C = [U[i] for i in Uc[:k]]
    C_old = None
    Z = None
    print('clen', len(C))
    while C_old is None or cost_km(C, C_old) > eps:
        C_old = C
        cIds, dists, _, clusters = getClusters(U, C_old)
        dists = sorted([(i,x) for i,x in enumerate(dists)], key = lambda x : x[1])
        Z = [U[x[0]] for x in dists[-z:]]
        print('z shape', len(Z))
        plotGraph(U, C_old, Z, "./Plots/KMeans_")
        X = [x[0] for x in dists[:-z]] # storing index of point in U
        print('u/z comp len', len(X))
        cNum = [0 for _ in range(k)]
        C = np.zeros((k, len(U[0])))
        for i in X:
            cNum[cIds[i]] += 1 # update no of points in cluster
            C[cIds[i]] = C[cIds[i]] + U[cIds[i]]
        for j in range(k):
            if cNum[j] != 0 : C[j] = C[j]/cNum[j]
            else : print('empty')
        print('costkm', cost_km(C, C_old))
    return C, Z


if __name__ == "__main__":
    random.seed(0)
    np.random.seed(0)
    U, y, C_, Z_, ids_ = make_data(5, 10, 10, 100, num_points = 10)
    C, Z = kmeans_(U, 3, 5)
    loss(U, C, Z, C_, Z_, ids_)
    plotGraph(U, C, Z, "./Plots/KMeans_")

'''
4 -> make_data(5, 0, 10, 50)
3 -> make_data(5, 10, 10, 50)
2 -> make_data(5, 0, 8, 50)
1 -> make_data(5, 0, 10, 100)
'''