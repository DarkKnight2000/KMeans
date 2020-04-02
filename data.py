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

# Remove duplicate data points if any
def removeDups(U, ys):
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
        print(mat)
        temp_X, temp_Y = removeDups(mat['X'], mat['y'])
        pickle.dump(temp_X, open('./Datasets/train_X.pkl', 'wb'))
        pickle.dump(temp_Y, open('./Datasets/train_y.pkl', 'wb'))

    return temp_X, temp_Y

def make_data(num_outliers, clus_sep, clus_var, cube_side, num_clus = 3, num_points = 30):
    C = [[random.uniform(clus_sep, cube_side), random.uniform(clus_sep, cube_side)], [random.uniform(clus_sep, -cube_side), random.uniform(clus_sep, cube_side)], [random.uniform(clus_sep, cube_side), random.uniform(clus_sep, -cube_side)]]
    U = []
    for i in range(num_clus):
        if len(U) != 0 : U = np.vstack((U, np.array([np.random.normal(C[i][0], clus_var, num_points), np.random.normal(C[i][1], clus_var, num_points)]).T))
        else : U = np.array([np.random.normal(C[i][0], clus_var, num_points), np.random.normal(C[i][1], clus_var, num_points)]).T
        # U.append(np.array([np.random.normal(C[i][0], clus_var, num_points), np.random.normal(C[i][1], clus_var, num_points)]).T)
        print(np.array(U).shape)
    U = np.vstack((U, [[random.uniform(-2*cube_side, 2*cube_side), random.uniform(-2*cube_side, 2*cube_side)] for _ in range(num_outliers)]))
    # print(U[0])
    print(np.array(U).shape)
    # print(U[:-num_outliers].T[0])
    ids, dists, _, _ = getClusters(U, C)
    print(ids[0])
    print(dists[0])
    dists = sorted([(i,x) for i,x in enumerate(dists)], key = lambda x : x[1])
    y = [0 for _ in range(len(U))]
    Z = []
    for x in dists[-num_outliers:]:
        y[x[0]] = 1
        Z.append(U[x[0]])
    Z = np.array(Z)

    # plotGraph(U, C, Z, "./Plots/SynthData")

    # print(dists)
    return U, y, C, Z, ids

def plotGraph(U, C, Z, foldername, filename = "trail11.png"):
    plt.plot(np.array(U).T[0], np.array(U).T[1], 'bo')
    plt.plot(np.array(C).T[0], np.array(C).T[1], 'ro')
    if Z is not None: plt.plot(np.array(Z).T[0], np.array(Z).T[1], 'go')
    plt.savefig(f"{foldername}/{filename}")
    plt.show()


if __name__ == "__main__":
    U,_,C,Z,_ = make_data(5, 0, 10, 100)
    plotGraph(U, C, Z, "./Plots/SynthData", "A1.png")