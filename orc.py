from scipy.cluster import vq
from lsalgo import d, sq_dist
import numpy as np

def getClusters(U, C):
    ids = []
    dists = []
    d_max = np.inf
    for x in U:
        cds = list(map(lambda c : sq_dist(x, c), C))
        ids.append(np.argmin(cds)) # gives index of nearest cluster center
        dists.append(cds[ids[-1]]) # get distance at that index, to return distances to nearest centers
        if d_max == np.inf or d_max < dists[-1]:
            d_max = dists[-1]
    return ids, np.array(dists), d_max

'''
    U -> Set of points to cluster
    I -> no of iterations to run
    T -> Threshold to classify as outlier
    return Centers, Outliers
'''
def orc(U, k, I, T, iniIters = 3):
    # Running kmeans multiple times to initialise C
    C = None
    Z = [] # outliers
    dist = np.inf
    C_ = None
    dist_ = np.Inf
    for _ in range(iniIters):
        C_, dist_ = vq.kmeans(vq.whiten(U), k)
        if dist_ < dist:
            C = C_
            dist = dist_

    for _ in range(I):
        cIds, cDists, d_max = getClusters(U, C)
        o_i = cDists/d_max
        U_ = []
        for i in range(len(U)):
            if o_i[i] <= T:
                U_.append(U[i])
            else:
                Z.append(U[i])
        U = U_
        C, _ = vq.kmeans(U, C)

    return list(C), Z

