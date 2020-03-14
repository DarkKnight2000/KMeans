import random
import numpy as np
import scipy.io
from sklearn.model_selection import train_test_split
import pickle
import sys

random.seed(0)

eps = 0.0001
load_data = True

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
                # print('1 ', len(C[:j]))
                # print('2 ', len(C[j+1:]))
                # print('3 ', len([u]))
                temp = [*C[:j],  *C[(j+1):], u]
                # print('c- shape in iter ', len(temp), ' for j ', j)
                c1 = cost_km(temp, U)
                if d(u, C) != 0 and c1 < cost_km(C_, U):
                    #print('changed--')
                    C_ = temp
                    #print('C- shape-- ',len(C_),',',len(C_[0]))
                    changed = True
        if not changed: break
        C = C_
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
    while alpha < 0 or (alpha*(1 - (eps/k))) > cost(C, Z, U):
        alpha = cost(C, Z, U)
        print('alpha', alpha)
        # {(i) local search with no outliers}
        C = LS(Complement(U, Z), C, k)
        #print('C shape ',len(C),',',len(C[0]))
        if len(C) != k:
            print('error in c')
            sys.exit(1)
        C_ = C # Copy for C
        Z_ = Z # Copy for Z

        # {(ii) cost of discarding z additional outliers}
        # temp = outliers(C, Z, U, z)
        # if cost(C, Z, U)*(1 - (eps/k)) > cost(C, Z + temp, U):
        #     Z_ = Z + temp

        # {(iii) for each center and non-center, perform a swap and discard additional outliers}
        for u in U:
            for i, v in enumerate(C):
                temp = C[:i] + C[i+1:] + [u]
                #print('temp', len(temp))
                if len(temp) != len(C):
                    print('error2')
                    sys.exit(1)
                if cost(C[:i] + C[i+1:] + [u], Z + temp, U) < cost(C_, Z_, U):
                    C_ = C[:i] + C[i+1:] + [u]
                    Z_ = Z + outliers(C[:i] + C[i+1:] + [u], [], U, z)

        # {update the solution allowing additional outliers if the solution value improved significantly}
        if cost(C, Z, U)*(1 - (eps/k)) > cost(C_, Z_, U):
            C =C_
            Z = Z_

        if len(C) != k:
            print('error in c')
            sys.exit(1)
    return C, Z


if load_data:
    temp_X = pickle.load(open('./Datasets/train_X.pkl', 'rb'))
    temp_Y = pickle.load(open('./Datasets/train_y.pkl', 'rb'))
else:
    mat = scipy.io.loadmat('./Datasets/cardio.mat')
    U = mat['X']
    temp_X = []
    temp_Y = []
    # removing duplicates
    for i,a in enumerate(U):
        f = False
        for j,b in enumerate(U):
            if i != j and sq_dist(a,b) == 0:
                # print('error')
                # print(a)
                # print(b)
                f = True
        if not f:
            temp_X.append(a)
            temp_Y.append(mat['y'][i])
    pickle.dump(temp_X, open('./Datasets/train_X.pkl', 'wb'))
    pickle.dump(temp_Y, open('./Datasets/train_y.pkl', 'wb'))
print(len(temp_X))


# X_train, X_test, y_train, y_test = train_test_split(np.array(temp_X), np.array(temp_Y), test_size=0.33, random_state=42)
# print(X_test.shape)


U = temp_X[:100]
# print(temp_Y)
# print(d(U[0], U))
# cost_km(U, U)
# LS_outlier(U, 1, 1)
print('u shape ', len(U),',',len(U[0]))
print(U[0][0])
print(U[1][0])
print(U[2][0])
# print(LS(U, [U[0]], 1)[0])
# print(cost_km([U[1]], U))
C, Z = (LS_outlier(U, 2, 10))

loss = 0
for i, x in enumerate(temp_X):
    if (d(x, Z) == 0) and temp_Y[i][0] == 0:
        loss += 1

print('loss : ', loss)