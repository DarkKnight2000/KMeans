import pickle
import random
import scipy.io
import numpy as np
from lsalgo import sq_dist, Complement, LS_outlier
from orc import getClusters
import matplotlib.pyplot as plt
import time


random.seed(0)
np.random.seed(0)


def removeDups(U, ys):
    '''
        Remove duplicate data points if any
    '''
    temp_X = []
    temp_Y = []

    # removing duplicates
    for i,a in enumerate(U):
        f = False
        for j,b in enumerate(U):
            if i != j and sq_dist(a,b) == 0:
                f = True
                break
        if not f:
            temp_X.append(a)
            temp_Y.append(ys[i])
    return temp_X, temp_Y


def load_file(load_data):
    if load_data:
        temp_X = pickle.load(open('./Datasets/train_X.pkl', 'rb'))
        temp_Y = pickle.load(open('./Datasets/train_y.pkl', 'rb'))
    else:
        mat = scipy.io.loadmat('./Datasets/cardio.mat')
        temp_X, temp_Y = removeDups(mat['X'], mat['y'])
        pickle.dump(temp_X, open('./Datasets/train_X.pkl', 'wb'))
        pickle.dump(temp_Y, open('./Datasets/train_y.pkl', 'wb'))

    return temp_X, temp_Y

def make_data(num_outliers, clus_sep, clus_var, cube_side, num_clus = 3, num_points = 30):
    '''
        Creates the Data (Synthetic Data)
    '''
    
    C = [[random.uniform(clus_sep, cube_side), random.uniform(clus_sep, cube_side)], [random.uniform(clus_sep, -cube_side), random.uniform(clus_sep, cube_side)], [random.uniform(clus_sep, cube_side), random.uniform(clus_sep, -cube_side)]]
    U = []
    
    for i in range(num_clus):
        if len(U) != 0: 
            U = np.vstack((U, np.array([np.random.normal(C[i][0], clus_var, num_points), np.random.normal(C[i][1], clus_var, num_points)]).T))
        else: 
            U = np.array([np.random.normal(C[i][0], clus_var, num_points), np.random.normal(C[i][1], clus_var, num_points)]).T

    U = np.vstack((U, [[random.uniform(-2*cube_side, 2*cube_side), random.uniform(-2*cube_side, 2*cube_side)] for _ in range(num_outliers)]))
    ids, dists, _, _ = getClusters(U, C)
    dists = sorted([(i,x) for i,x in enumerate(dists)], key = lambda x : x[1])
    y = [0 for _ in range(len(U))]
    Z = []
    for x in dists[-num_outliers:]:
        y[x[0]] = 1
        Z.append(U[x[0]])
    Z = np.array(Z)

    return U, y, C, Z, ids

def plotGraph(U, C, Z, foldername, filename = "trail11.png"):

    plt.scatter(np.array(Complement(U, Z)).T[0], np.array(Complement(U, Z)).T[1],marker = 'o', color = 'k', label = "Clusters")
    plt.scatter(np.array(C).T[0], np.array(C).T[1],marker = '.', color = 'b', s = 300, label = "Centroids")
    if Z is not None: 
        plt.scatter(np.array(Z).T[0], np.array(Z).T[1], marker = "x", color = 'r', label = "Outliers")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    U,_,C,Z,_ = make_data(5, 10, 10, 50)    
    plotGraph(U, C, Z, "./Plots/SynthData", "A1.png")