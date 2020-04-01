import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image
import pandas as pd

Clusters = {}

def assign_cluster(A, B) :
    index = np.zeros(A.shape[1])
    for j in range (A.shape[1]) :
        C = np.zeros((d,B.shape[1]))
        C = B - A[:,j:j+1]
        C = np.square(C)
        index[j] = np.argmin(np.sum(C, axis = 0))
    return index

def update_clusters(A, I, z) :
    m, n = A.shape
    clus = np.zeros((m, z))
    for i in range(z) :
        Clusters.update({'cluster-'+str(i+1) : A[:,I == i]})
        clus[:,i] = np.mean(X[:,I == i], axis = 1)
    return clus

def dis(A, B, e,i) :
    C = A - B
    C = np.square(C)
    C = np.sum(C)
    print("Error at Iteration " + str(i) + " : ", C,"\n")
    if (C > e) :
        return 1
    return 0


def k_means(X) :
    curr_clusters = X[:,:K]
    prev_clusters = np.zeros((d,K))
    Iteration = 0
    while (dis(curr_clusters,prev_clusters,E,Iteration)) :
        Clusters.clear()
        index = assign_cluster(X, curr_clusters)
        prev_clusters = curr_clusters
        curr_clusters = update_clusters(X, index, K)
        Iteration += 1
    print("final_clusters : \n",curr_clusters,"\n")
    i=0
    mu = np.zeros((d,K))
    sigma = np.zeros((K,d,d))
    for x in Clusters:
        print(x,":\n",Clusters[x],"\n")
        mu[:,i] = np.sum(Clusters[x], axis = 1)/n
        sigma[i,:,:] = np.cov(Clusters[x])
        i+=1
    return mu, sigma




case = int(input("For Image input '1' -- For Iris Data Set input '2' : "))

if (case == 1) :
    im = (Image.open("img-1.jpeg"))
    pixels = np.array(list(im.getdata())).T
    X = pixels
    print(X)
    K = int(input("Enter the number of clusters : "))
    E = float(input("Enter the Threshold : "))
    d = X.shape[0]
    n = X.shape[1]
    k_means(X)
    fig = plt.figure(1)
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(pixels[0,:],pixels[1,:],pixels[2,:])
    fig2 = plt.figure(2)
    ax2 = fig2.add_subplot(111, projection='3d')
    for x in Clusters:
        c = '#%02X%02X%02X' % (np.random.randint(0,255),np.random.randint(0,255),np.random.randint(0,255))
        ax2.scatter(Clusters[x][0,:],Clusters[x][1,:],Clusters[x][2,:],color = c)
    plt.show()



elif (case == 2) :
    dataset = pd.read_csv('iris.csv')
    X = dataset.iloc[:, [0, 1, 2, 3]].values.T
    K = int(input("Enter the number of clusters : "))
    E = float(input("Enter the Threshold : "))
    d = X.shape[0]
    n = X.shape[1]
    k_means(X)
    for name,value in Clusters.items():
        c = '#%02X%02X%02X' % (np.random.randint(0,255),np.random.randint(0,255),np.random.randint(0,255))
        plt.scatter(value[1,:],value[2,:],s = 70, color = c, label = name)
    plt.show()
