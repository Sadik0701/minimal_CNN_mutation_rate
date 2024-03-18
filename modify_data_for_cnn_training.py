import numpy as np
import keras
from sklearn.neighbors import NearestNeighbors
import numpy

max_len = 0

def unstandardize(y, mean, std):
    return y*std+mean

def sort_min_diff(amat):
    mb = NearestNeighbors(len(amat), metric='manhattan').fit(amat)
    v = mb.kneighbors(amat)
    smallest = numpy.argmin(v[0].sum(axis=1))
    return amat[v[1][smallest]]

my_input_test = np.load('Data/Simulations/sim_n_40_rep_10000_rho_0_theta_random-100.npz', allow_pickle=True)
G_test = []
theta = []
my_multi_theta = []
multi_theta = []
i = 0
my_G_test = []


for theta in zip(my_input_test["multi_theta"]):
    my_multi_theta.append(theta)

for i in range(0,10000):
    multi_theta.append(my_multi_theta[i][0])
#print(multi_theta)
print("len theta:", len(multi_theta))

for G in zip(my_input_test["multi_G"]):
    #print(G)
    my_G_test.append(G)
    #print(len(G[0]))
    if len(G[0]) > max_len:
        max_len = len(G[0])
    #my_theta.append(ytest)
    #i = i +1
    #print(i)#

print("max_len", max_len)

my_zwischen_G = []#

for i in range(0,len(multi_theta)):
    my_G_test[i] = numpy.array(my_G_test[i])
    #print(my_G_test[i][0].shape[0])
    if my_G_test[i][0].shape[0] < (max_len +1):
        A = numpy.zeros(((max_len - my_G_test[i][0].shape[0]), 40))
        #print(A.shape)
        B = numpy.concatenate((my_G_test[i][0], A))
        my_zwischen_G.append(B)
    elif i%1000==0:
        print(i)
    else:
        C = my_G_test[i][0][:max_len,:]
        print("zu lang:", C.shape)
        my_zwischen_G.append(C)

G_filled = numpy.stack([my_zwischen_G[l] for l in range(0, len(multi_theta))])
print(G_filled.shape)

multi_G_test = G_filled[:2000]
multi_G_train = G_filled[2000:]
multi_theta_test = multi_theta[:2000]
multi_theta_train = multi_theta[2000:]

print(multi_G_test.shape)
print(multi_G_train.shape)
print(len(multi_theta_test))
print(len(multi_theta_train))

filename1 = ("rho_0_theta_random_sim_no_seed_rep_10000")

numpy.savez_compressed(filename1, xtest=multi_G_test, xtrain=multi_G_train, ytest=multi_theta_test, ytrain=multi_theta_train)
