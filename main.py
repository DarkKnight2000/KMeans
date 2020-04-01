import pickle
from lsalgo import sq_dist, LS_outlier, d, Complement
from orc import orc
import random
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import time

random.seed(0)

load_data = True


## Setting keys to run only required algos
LSAlgo, ORC = 1, 2
RunAlgos = [2]

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
        if not f:
            temp_X.append(a)
            temp_Y.append(ys[i])
    return temp_X, temp_Y

if load_data:
    temp_X = pickle.load(open('./Datasets/train_X.pkl', 'rb'))
    temp_Y = pickle.load(open('./Datasets/train_y.pkl', 'rb'))
else:
    mat = scipy.io.loadmat('./Datasets/cardio.mat')
    temp_X, temp_Y = removeDups(mat['X'], mat['y'])
    pickle.dump(temp_X, open('./Datasets/train_X.pkl', 'wb'))
    pickle.dump(temp_Y, open('./Datasets/train_y.pkl', 'wb'))
print(len(temp_X))


# X_train, X_test, y_train, y_test = train_test_split(np.array(temp_X), np.array(temp_Y), test_size=0.33, random_state=42)
# print(X_test.shape)

'''
    Getting data ready
'''
U = temp_X[:100]
U = [u[3:5] for u in U]
U, y = removeDups(U, temp_Y)
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


if LSAlgo in RunAlgos:
    '''
        Running LS Algo and getting centers
    '''
    C, Z = (LS_outlier(U, 4, 5))

    # loss = 0
    # for i, x in enumerate(U):
    #     if (d(x, Z) == 0) and y[i][0] == 0:
    #         loss += 1

    # print('loss : ', loss)

    print('No of centers : ', len(C))
    print('No of outliers detected : ', len(Z))

    '''
        Plotting centers and outliers by LS Algo
    '''
    plt.plot(np.array(Complement(U, C+Z)).T[0], np.array(Complement(U, C+Z)).T[1], 'bo')
    plt.plot(np.array(C).T[0], np.array(C).T[1], 'ro')
    plt.plot(np.array(Z).T[0], np.array(Z).T[1], 'go')
    plt.savefig(f"./Plots/LSAlgoPlots/trail{time.time()}.png")
    plt.show()

if ORC in RunAlgos:
    '''
        Running ORC Algo and getting centers
    '''
    C, Z = (orc(U, 4, 5, 0.95))

    # loss = 0
    # for i, x in enumerate(U):
    #     if (d(x, Z) == 0) and y[i][0] == 0:
    #         loss += 1

    # print('loss : ', loss)

    print('No of centers : ', len(C))
    print('No of outliers detected : ', len(Z))

    '''
        Plotting centers and outliers by ORC Algo
    '''
    plt.plot(np.array(Complement(U, C+Z)).T[0], np.array(Complement(U, C+Z)).T[1], 'bo')
    plt.plot(np.array(C).T[0], np.array(C).T[1], 'ro')
    plt.plot(np.array(Z).T[0], np.array(Z).T[1], 'go')
    plt.savefig(f"./Plots/ORCAlgoPlots/trail{time.time()}.png")
    plt.show()