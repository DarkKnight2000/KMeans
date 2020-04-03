import pickle
from lsalgo import sq_dist, LS_outlier, d, Complement, cost, cost_km
from orc import orc, getClusters
from data import removeDups, load_file, plotGraph, make_data
from kmeans_minus import kmeans_minus
import random
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import timeit

random.seed(0)
np.random.seed(0)

load_data = True
real_data = False

# Setting keys to run only required algos
LSAlgo, ORC, KMeans_ = 1, 2, 3
RunAlgos = [2]


def loss(U, C, Z, C_, Z_, cIds_):
    '''
        U : Total Data
        C : Cluster Centers
        Z : Detected Outliers
        Z_ : Actual Outliers
        C_ : Detected Cluster Centers
        cIds_ : Index of the Cluster point belongs 
    '''

    # precision = no of points of Z_ in Z
    common = 0
    for z_ in Z_:
        if d(z_, Z) == 0: 
            common += 1

    print('Precision : ', common/len(Z))
    print('Recall : ', common/len(Z_))

    ids, dists, _, _ = getClusters(U, C)
    err = 0
    costval = 0
    err_o = 0
    for i in range(len(U)):

        # for non-outliers
        if d(U[i], Z_) != 0:
            err += np.abs(pow(sq_dist(C[ids[i]], C_[cIds_[i]]), 0.5))
            costval += sq_dist(C[ids[i]], U[i])
        else:
            err_o += np.abs(pow(sq_dist(C[ids[i]], C_[cIds_[i]]), 0.5))

    print('Distance ratio : ', err/(len(U) - len(Z)))
    print('Distance ratio with outliers : ', err_o/len(Z))
    print('Cost : ', costval/len(U))



def blindLoss(X, y, C, Z):
    '''
        X : Total Data
        y : Cluster Centers
        C : Detected Cluster Centers
        Z : Detected Outliers
    '''

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


if __name__ == "__main__":

    # Loading the existing data
    if real_data:
        temp_X, temp_Y = load_file(load_data)
        random.shuffle(temp_X)
        random.shuffle(temp_Y)

        U, y = removeDups(temp_X, temp_Y)

    # Synthetic Data
    else:
        U, y, C_, Z_, ids_ = make_data(5, 0, 8, 50)


    # # X_train, X_test, y_train, y_test = train_test_split(np.array(temp_X), np.array(temp_Y), test_size=0.33, random_state=42)
    # # print(X_test.shape)

    # # data is finally in U and labels in y
    # print('u shape ', len(U),',',len(U[0]))
    # print(U[0][0])
    # print(U[1][0])
    # print(U[2][0])
    # # print(LS(U, [U[0]], 1)[0])
    # # print(cost_km([U[1]], U))


    if LSAlgo in RunAlgos:
        '''
            Running LS Algo and getting Centers & Outliers
        '''

        Uc = U.copy()
        in_ = timeit.default_timer()
        C, Z = (LS_outlier(U, 3, 5))
        out_ = timeit.default_timer()

        print('Runtime : ', out_ - in_)
        print('No of centers : ', len(C))
        print('No of outliers detected : ', len(Z))

        # Plotting centers and outliers by LS Algo
        plotGraph(U, C, Z, "./Plots/LSAlgoPlots")

        if real_data: 
            blindLoss(U, y, C, Z)
        else:
            loss(U, C, Z, C_, Z_, ids_)

    if ORC in RunAlgos:
        '''
            Running ORC Algo and getting Centers & Outliers
        '''

        in_ = timeit.default_timer()        
        C, Z = (orc(U, 3, 5, 0.95))
        out_ = timeit.default_timer()

        print('Runtime : ', out_ - in_)
        print('No of centers : ', len(C))
        print('No of outliers detected : ', len(Z))

        # Plotting centers and outliers by ORC Algo
        plotGraph(U, C, Z, "./Plots/ORCAlgoPlots")
        
        if real_data: 
            blindLoss(U, y, C, Z)
        else:
            loss(U, C, Z, C_, Z_, ids_)

    if KMeans_ in RunAlgos:
        '''
            Running KMeans-- Algo and getting Centers & Outliers
        '''

        in_ = timeit.default_timer()        
        C, Z = kmeans_minus(U, 3, 5)
        out_ = timeit.default_timer()

        print('Runtime : ', out_ - in_)

        # Plotting centers and outliers by KMeans-- Algo
        plotGraph(U, C, Z, "./Plots/KMeans_")

        if real_data: 
            blindLoss(U, y, C, Z)
        else:
            loss(U, C, Z, C_, Z_, ids_)