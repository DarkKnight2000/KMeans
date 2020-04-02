import pickle
from lsalgo import sq_dist, LS_outlier, d, Complement, cost, cost_km
from orc import orc, getClusters
from data import removeDups, load_file, plotGraph, make_data
import random
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import time

random.seed(0)
np.random.seed(0)

load_data = True
real_data = False

## Setting keys to run only required algos
LSAlgo, ORC = 1, 2
RunAlgos = [1]

# Calculating loss and stats
# Z -> calculated outliers
# Z_ -> actual outliers
def loss(U, C, Z, C_, Z_, cIds_):
    # precision = no of points of Z_ in Z
    common = 0
    for z_ in Z_:
        if d(z_, Z) == 0 : common += 1
    print('Precision : ', common/len(Z))
    print('Recall : ', common/len(Z_))

    ids, dists, _ = getClusters(U, C)
    err = 0
    costval = 0
    for i in range(len(U)):
        if d(U[i], Z_) != 0: #for non-outliers
            err += np.abs(pow(sq_dist(C[ids[i]], C_[cIds_[i]]), 0.5))
            costval += sq_dist(C[ids[i]], U[i])
    print('MAE : ', err)

    print('Cost : ', costval/len(U))

def blindLoss(X, y, C, Z):
    if len(X) != len(y):
        print('Input sizes not matching')
        return
    common = 0
    actual = 0
    for i in range(len(X)):
        if y[i] == 1:
            actual += 1
            if d(X[i], Z) == 0:
                common += 1
    print('No of actual outiers : ', actual)
    print('Precision : ', common/len(Z))
    print('Recall : ', common/actual)

    print('Cost : ', cost(C, X, Z))

if real_data:
    temp_X, temp_Y = load_file(load_data)
    random.shuffle(temp_X)
    random.shuffle(temp_Y)
    print(len(temp_X))
    '''
    Getting data ready
    '''
    # U = [u[1:3] for u in U]
    U, y = removeDups(temp_X[-150:-50], temp_Y[-150:-50])
    # print(temp_Y)
    # print(d(U[0], U))
    # cost_km(U, U)
    # LS_outlier(U, 1, 1)

else:
    U, y, C_, Z_, ids_ = make_data(5, 10, 10, 100)


# X_train, X_test, y_train, y_test = train_test_split(np.array(temp_X), np.array(temp_Y), test_size=0.33, random_state=42)
# print(X_test.shape)

# data is finally in U and labels in y
print('u shape ', len(U),',',len(U[0]))
print(U[0][0])
print(U[1][0])
print(U[2][0])
# print(LS(U, [U[0]], 1)[0])
# print(cost_km([U[1]], U))


if LSAlgo in RunAlgos:
    '''
        Running LS Algo and getting centers
    '''
    Uc = U[:]
    C, Z = (LS_outlier(U, 3, 5))

    print('No of centers : ', len(C))
    print('No of outliers detected : ', len(Z))
    print('i', len(U))
    for i in range(len(U)):
        if sq_dist(Uc[i], U[0]) != 0:
            print('error')
    '''
        Plotting centers and outliers by LS Algo
    '''
    plotGraph(U, C, Z, "./Plots/LSAlgoPlots")
    if real_data : blindLoss(U, y, C, Z)
    else :
        loss(U, C, Z, C_, Z_, ids_)

if ORC in RunAlgos:
    '''
        Running ORC Algo and getting centers
    '''
    C, Z = (orc(U, 3, 5, 0.95))

    print('No of centers : ', len(C))
    print('No of outliers detected : ', len(Z))

    '''
        Plotting centers and outliers by ORC Algo
    '''
    plotGraph(U, C, Z, "./Plots/ORCAlgoPlots")
    if real_data : blindLoss(U, y, C, Z)
    else :
        loss(U, C, Z, C_, Z_, ids_)