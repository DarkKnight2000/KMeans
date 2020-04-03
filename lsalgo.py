import random
import numpy as np
from sklearn.model_selection import train_test_split
import sys

def sq_dist(u, v):
    '''
        Returns eucledian distance between the points
    '''
    if(len(u) != len(v)):
        print('Diff shapes')
        return 0
    return np.sum((u-v)**2)

def d(v, S, key = "min"):
    '''
        Returns min of distances from v to all the points in S
    '''
    di = sq_dist(S[0],v)
    for x in S:
        if key == "min":
            di = min(di, sq_dist(x,v))
            if(di == 0): return 0
        else:
            di = max(di, sq_dist(x,v))
    return di


def cost_km(S, C, k = None, key = "min"):
    
    # min = true then get least distance points o.w max distance
    dists = list(map(lambda x: d(x, S), C))
    if k is None: 
        return np.sum(dists) # return sum of distances of points in C from S
    else:
        # return k farthest points from S in C
        x = sorted(zip(dists, C), key= lambda x : x[0])
        if key != "min": x = list(reversed(x))
        x = [a[1] for a in x[:k]]
        return x


def Complement(U, Z):
    '''
        Returns Complement of Z w.r.t U
    '''
    if Z == []: return U
    return [u for u in U if d(u,Z) != 0]


def cost(C, U, Z):
    '''
        Get sum of distances from S all points in U/C
    '''
    return cost_km(C, Complement(U, Z))

def outliers(S, C, U, k):
    '''
        S : farthest points from this set are calculated
        U/C : Points from this set are searched for the farthest points
        k : no. of farthest points required
    '''

    return cost_km(S, Complement(U, C), k = k, key = "max")


def LS(U, C, k, eps):
    '''
        U : Total Data
        C : Centers
    '''

    alpha = np.inf
    while (alpha*(1 - (eps/k)) > cost_km(C, U)):
        alpha = cost_km(C, U)
        C_ = C # Copy for C
        for i, u in enumerate(U): # Searching all non-centers to replace one of the centers
            for j, v in enumerate(C):
                if d(u, C) == 0: 
                    continue
                temp = [*C[:j],  *C[(j+1):], u]
                c1 = cost_km(temp, U)
                if c1 < cost_km(C_, U):
                    C_ = temp
        C = C_
    return C


def LS_outlier(U, k, z, eps = .00001):
    '''
        U : Total Data
        k : No of Centers
        z : No of Outlers
        Returns : C (Detected Centers) & Z (Detected Outliers)
    '''

    U_ = U.copy()
    random.shuffle(U_)
    C = U_[:k]
    
    Z = outliers(C, [], U, z)
    if len(Z) != z:
        print('error in z')
        sys.exit(1)

    alpha = np.inf
    while ((alpha*(1 - (eps/k))) > cost(C, U, Z)):
        alpha = cost(C, U, Z)

        # {(i) local search with no outliers}
        C = LS(Complement(U, Z), C, k, eps)
        if len(C) != k:
            print('error in c')
            sys.exit(1)
        C_ = C # Copy for C
        Z_ = Z # Copy for Z

        # {(ii) cost of discarding z additional outliers}
        temp = outliers(C, Z, U, z)
        if cost(C, U, Z)*(1 - (eps/k)) > cost(C, U, Z + temp):
            Z_ = Z + temp

        # {(iii) for each center and non-center, perform a swap and discard additional outliers}
        for u in U:
            for i, v in enumerate(C):
                if d(u, C) == 0: continue
                temp = C[:i] + C[i+1:] + [u]
                if len(temp) != len(C):
                    print('error2')
                    sys.exit(1)
                if cost(temp, U, Z + outliers(temp, Z, U, z)) < cost(C_, U, Z_):
                    C_ = C[:i] + C[i+1:] + [u]
                    Z_ = Z + outliers(temp, Z, U, z)


        # {update the solution allowing additional outliers if the solution value improved significantly}
        if cost(C, U, Z)*(1 - (eps/k)) > cost(C_, U, Z_):
            C =C_
            Z = Z_

        if len(C) != k:
            print('error in c')
            sys.exit(1)

    return C, Z