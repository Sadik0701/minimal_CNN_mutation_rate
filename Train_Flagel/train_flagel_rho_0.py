import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv1D, MaxPooling1D, AveragePooling1D,Conv2D,MaxPooling2D, AveragePooling2D
from random import choice
from matplotlib import pyplot as plt
from sklearn.neighbors import NearestNeighbors
from pprint import pprint

nreps, nepoch = 10,10

def sort_min_diff(amat):
    mb = NearestNeighbors(n_neighbors=len(amat), metric='manhattan').fit(amat)
    v = mb.kneighbors(amat)
    smallest = np.argmin(v[0].sum(axis=1))
    return amat[v[1][smallest]]

def transpose_shape(x):
    n = []
    for i in x: n.append(i.T)
    return np.array(n)

def standardize(y):
    return y.mean(), y.std(), (y-y.mean())/y.std()

def unstandardize(y, mean, std):
    return y*std+mean

a = np.load('../rho_0_theta_random_sim_no_seed_rep_10000.npz', allow_pickle=True)
xtrain, xtest, ytest, ytrain = [a[i] for i in ['xtrain', 'xtest', 'ytest', 'ytrain']]
print(np.shape(xtrain))
print(xtrain.shape)
n_seg_sites, n_indv = xtrain[1].shape
print(n_seg_sites, n_indv)
xtest_untransposed, xtrain_untransposed = map(transpose_shape, [xtest, xtrain])
ytest_mean, ytest_std, ytest = standardize(ytest)
ytrain_mean, ytrain_std, ytrain = standardize(ytrain)
print(ytest_mean, ytest_std)

all_out = {'not_transposed':[], 'transpose':[], 'sort_and_transpose':[]}
 

rtrain = np.array([sort_min_diff(i.T).T for i in xtrain])
rtest = np.array([sort_min_diff(i.T).T for i in xtest])


for i in range(nreps):
    model = Sequential()
    model.add(Conv2D(filters=64, kernel_size=(n_indv,1), activation='relu',input_shape=(n_seg_sites,n_indv,1),padding='valid'))
    """
    model.add(Conv1D(64, kernel_size=2,
                     activation='relu',
                     input_shape=(n_seg_sites, n_indv)))
    model.add(Conv1D(64, kernel_size=2, activation='relu'))
  
"""
    model.add(AveragePooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(64, activation='relu', kernel_initializer='normal'))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu', kernel_initializer='normal'))
    model.add(Dropout(0.5))
    model.add(Dense(1, kernel_initializer='normal'))
    model.compile(loss='mean_squared_error', optimizer='adam')
    #model.summary()
    pred = model.predict(rtest)
    rmse = np.mean([(iii-jjj)**2 for iii,jjj in zip(unstandardize(ytest, ytest_mean, ytest_std), unstandardize(pred, ytest_mean, ytest_std))])**0.5
    f = [rmse]
    print(i, f[0])
    if i == 0:
        print("shape xtrain:", np.shape(xtrain))
        print("mean:", ytest_mean, "std:", ytest_std)
    for j in range(nepoch):    
        model.fit(rtrain, ytrain, batch_size=32,
                  epochs=1, verbose=0, validation_data=(rtest, ytest))
        pred = model.predict(rtest)
        rmse = np.mean([(iii-jjj)**2 for iii,jjj in zip(unstandardize(ytest, ytest_mean, ytest_std), unstandardize(pred, ytest_mean, ytest_std))])**0.5
        print( i,j, rmse)
        f.append(rmse)
    all_out['sort_and_transpose'].append(f)
    filename = ("Flagel_CNN_rho_0_theta_0_100_rep_10000_no_seed_pass_nx1_" +str(i))
    model.save(filename)
pprint( all_out )
print("done")
