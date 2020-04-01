import random
import numpy as np
from sklearn.model_selection import train_test_split
import sys

def sq_dist(u, v):
    if(len(u) != len(v)):
        print('Diff shapes')
        return 0
    #print(u[0], v[0], np.sum((u-v)**2))
    return np.sum((u-v)**2)

def d(v, S, key = "min"):
    di = sq_dist(S[0],v)
    for x in S:
        if key == "min":
            di = min(di, sq_dist(x,v))
            if(di == 0): return 0
        else:
            di = max(di, sq_dist(x,v))
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
def LS(U, C, k, eps):

    print('C shape-- ',len(C),',',len(C[0]))
    print('U shape-- ',len(U),',',len(U[0]))
    print('a--',cost_km(C, U))
    alpha = -1
    while alpha < 0 or (alpha*(1 - (eps/k)) > cost_km(C, U)):
        alpha = cost_km(C, U)
        C_ = C # Copy for C
        changed = False
        for i, u in enumerate(U): # Searching all non-centers to replace one of the centers
            for j, v in enumerate(C):
                if d(u, C) == 0: continue
                # print('1 ', len(C[:j]))
                # print('2 ', len(C[j+1:]))
                # print('3 ', len([u]))
                temp = [*C[:j],  *C[(j+1):], u]
                # print('c- shape in iter ', len(temp), ' for j ', j)
                c1 = cost_km(temp, U)
                if c1 < cost_km(C_, U):
                    #print('changed--')
                    C_ = temp
                    #print('C- shape-- ',len(C_),',',len(C_[0]))
                    changed = True
        if not changed: break
        C = C_
    return C


def LS_outlier(U, k, z, eps = 4):
    random.shuffle(U)
    print('u shape ', len(U),',',len(U[0]))
    C = U[:k]
    C_ = U[k:]
    print('C shape ',len(C),',',len(C[0]))
    print('C- shape ',len(C_),',',len(C_[0]))
    print('cost of c, c_', cost(C, C_, U))
    
    Z = outliers(C, [], U, z)
    if len(Z) != z:
        print('error in z')
        sys.exit(1)
    # print(Z[0])
    # print(U[1][0])
    # print(Z[0][0]-U[1][0])
    print('z shape',len(Z))
    print('U shape ',len(U),',',len(U[0]))
    print('comp shape ', len(Complement(U, Z)))

    alpha = -1
    while (alpha < 0 or (alpha*(1 - (eps/k))) > cost(C, Z, U)) and len(Z)+k+z < len(U) :#FIXME: remove last condition
        alpha = cost(C, Z, U)
        print('alpha', alpha)
        # {(i) local search with no outliers}
        print('U Z comp shape ',len(Complement(U, Z)),',',len(Complement(U, Z)[0]))
        C = LS(Complement(U, Z), C, k, eps)
        #print('C shape ',len(C),',',len(C[0]))
        if len(C) != k:
            print('error in c')
            sys.exit(1)
        C_ = C # Copy for C
        Z_ = Z # Copy for Z

        # {(ii) cost of discarding z additional outliers}
        temp = outliers(C, Z, U, z)
        if cost(C, Z, U)*(1 - (eps/k)) > cost(C, Z + temp, U):
            Z_ = Z + temp

        # {(iii) for each center and non-center, perform a swap and discard additional outliers}
        for u in U:
            for i, v in enumerate(C):
                if d(u, C) == 0: continue
                temp = C[:i] + C[i+1:] + [u]
                #print('temp', len(temp))
                if len(temp) != len(C):
                    print('error2')
                    sys.exit(1)
                if cost(temp, Z + outliers(temp, Z, U, z), U) < cost(C_, Z_, U):
                    C_ = C[:i] + C[i+1:] + [u]
                    Z_ = Z + outliers(temp, Z, U, z)
                    print('changed c_ ', len(C_),',',len(C_[0]),', ___ ',len(Z_),',',len(Z_[0]))
                    print('U shape ',len(U),',',len(U[0]))


        # {update the solution allowing additional outliers if the solution value improved significantly}
        if cost(C, Z, U)*(1 - (eps/k)) > cost(C_, Z_, U):
            C =C_
            Z = Z_
        print('looped')

        print('u shape ', len(U),',',len(U[0]))
        print('C shape ',len(C),',',len(C[0]))
        print('Z shape ',len(Z),',',len(Z[0]))
        print('U Z comp shape ',len(Complement(U, Z)),',',len(Complement(U, Z)[0]))
        print('pres cost ', cost(C, Z, U))

        if len(C) != k:
            print('error in c')
            sys.exit(1)
    return C, Z