import random
import numpy as np
import scipy.io
from sklearn.model_selection import train_test_split

random.seed(0)

mat = scipy.io.loadmat('./Datasets/cardio.mat')
X_train, X_test, y_train, y_test = train_test_split(mat['X'], mat['y'], test_size=0.33, random_state=42)
print(X_test.shape)

eps = 0.0001

def sq_dist(u, v):
    if(len(u) != len(v)):
        print('Diff shapes')
        return 0
    #print(u[0], v[0], np.sum((u-v)**2))
    return np.sum((u-v)**2)

def d(v, S):
    di = sq_dist(S[0],v)
    for x in S:
        di = min(di, sq_dist(x,v))
        if(di == 0): return 0
    return di

'''
Distances from S to all points in C
'''
def cost_km(S, C, k = None, min = False):
# min = true then get least distance points o.w max distance
    dists = list(map(lambda x: d(x, S), C))
    #print('x ', dists)
    if k is None : return np.sum(dists) # return sum of distances of points in C from S
    else:
        # return k farthest points from S in C
        x = sorted(zip(dists, C), key= lambda x : x[0], reverse = (not min))[:k]
        x = [a[1] for a in x]
        return x

# Complement of Z with respect to U
def Complement(U, Z):
    if Z == []: return U
    return [u for u in U if d(u,Z) != 0]

'''
Get sum of distances from S all points in U/C
'''
def cost(S, C, U):
    return cost_km(S, Complement(U, C))

'''
S -> farthest points from this set are calculated
U/C -> Points from this set are searched for the farthest points
k -> no. of farthest points required
'''
def outliers(S, C, U, k):
    return cost_km(S, Complement(U, C), k)

# U is all points
# C is the set of centers
def LS(U, C, k):
    U_ = Complement(U, C) # Always equal to U/C

    print('C shape-- ',len(C),',',len(C[0]))
    print('C- shape-- ',len(U_),',',len(U_[0]))
    print('U shape-- ',len(U),',',len(U[0]))
    print('cost of c, c_---', cost(C, U_, U))
    print('a--',cost_km(C, U))
    print('b--',cost_km(C, U_))
    alpha = -1
    while alpha < 0 or (alpha*(1 - (eps/k)) > cost_km(C, U_)):
        alpha = cost_km(C, U_)
        C_ = C # Copy for C
        U_2 = U_ # Copy for U_
        changed = False
        for i, u in enumerate(U_): # Searching all non-centers to replace one of the centers
            for j, v in enumerate(C):
                if cost_km(C[:j] + C[j+1:] + [u], U_[:i] + U_[i+1:] + [v]) < cost_km(C_, U_2):
                    print('changed--')
                    print('C- shape-- ',len(C_),',',len(C_[0]))
                    changed = True
                    C_ = C[:j] + C[j+1:] + [u]
                    U_2 = U_[:i] + U_[i+1:] + [v]
        if not changed: break
        C = C_
        U_ = U_2
    return C


def LS_outlier(U, k, z):
    random.shuffle(U)
    print('u shape ', len(U),',',len(U[0]))
    C = U[:k]
    C_ = U[k:]
    print('C shape ',len(C),',',len(C[0]))
    print('C- shape ',len(C_),',',len(C_[0]))
    print('cost of c, c_', cost(C, C_, U))
    
    Z = outliers(C, [], U, z)
    # print(Z[0])
    # print(U[1][0])
    # print(Z[0][0]-U[1][0])
    print('z shape',len(Z))

    alpha = -1
    while alpha < 0 or (alpha*(1 - (eps/k))):
        alpha = cost(C, Z, U)

        # {(i) local search with no outliers}
        C = LS(Complement(U, Z), C, k)
        print('C shape ',len(C),',',len(C[0]))

        C_ = C # Copy for C
        Z_ = Z # Copy for Z

        # {(ii) cost of discarding z additional outliers}
        temp = outliers(C, Z, U, z)
        if cost(C, Z, U)*(1 - (eps/k)) > cost(C, Z + temp, U):
            Z_ = Z + temp

        # {(iii) for each center and non-center, perform a swap and discard additional outliers}
        for u in U:
            for i, v in enumerate(C):
                temp = outliers(C[:i] + C[i+1:] + [u], [], U, z)
                if cost(C[:i] + C[i+1:] + [u], Z + temp, U) < cost(C_, Z_, U):
                    C_ = C[:i] + C[i+1:] + [u]
                    Z_ = Z + outliers(C[:i] + C[i+1:] + [u], [], U, z)

        # {update the solution allowing additional outliers if the solution value improved significantly}
        if cost(C, Z, U)*(1 - (eps/k)) > cost(C_, Z_, U):
            C =C_
            Z = Z_
    return C, Z


U = X_train
# print(s)
# print(d(U[0], U))
# cost_km(U, U)
# LS_outlier(U, 1, 1)
print('u shape ', len(U),',',len(U[0]))
print(U[0][0])
print(U[1][0])
print(U[2][0])
# print(LS(U, [U[0]], 1)[0])
# print(cost_km([U[1]], U))

print(len(LS_outlier(U, 2, 1)[1]))