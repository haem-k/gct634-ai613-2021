# GCT634 (2018) HW1
#
# Mar-18-2018: initial version
#
# Juhan Nam
#

import sys
import os
import numpy as np
import librosa
from feature_summary import *

from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.mixture import GaussianMixture

def train_model(train_X, train_Y, valid_X, valid_Y, hyper_param1, classifier='sgd'):

    # Choose a classifier (here, linear SVM)
    if classifier == 'sgd':
        clf = SGDClassifier(verbose=0, loss="hinge", alpha=hyper_param1, max_iter=1000, penalty="l2", random_state=0)
    elif classifier == 'knn':
        clf = KNeighborsClassifier(n_neighbors=5)
    elif classifier == 'svc':
        clf = SVC()
    # elif classifier == 'gmm':
    #     clf = GaussianMixture(n_components=10)

    # train
    clf.fit(train_X, train_Y)

    # validation
    valid_Y_hat = clf.predict(valid_X)

    accuracy = np.sum((valid_Y_hat == valid_Y))/300.0*100.0
    print(f'alpha = {hyper_param1}, validation accuracy = {accuracy}%')
    
    return clf, accuracy

if __name__ == '__main__':

    # # load data 
    # train_X = mean_mfcc('train')
    # valid_X = mean_mfcc('valid')

    # load mfcc delta
    train_dmfcc = delta_mfcc(dataset='train')
    valid_dmfcc = delta_mfcc(dataset='valid')

    # # load mfcc double delta
    # train_dd = double_delta_mfcc(dataset='train')
    # valid_dd = double_delta_mfcc(dataset='valid')

    # load rms
    train_rms = load_rms('train')
    valid_rms = load_rms('valid')

    # # load rms delta
    # train_rms_delta = delta_rms('train')
    # valid_rms_delta = delta_rms('valid')

    # concat features
    train_X = np.concatenate((train_dmfcc, train_rms), axis=0)
    valid_X = np.concatenate((valid_dmfcc, valid_rms), axis=0)


    print(f"Input feature dimension: {train_X.shape}")

    # label generation
    cls = np.array([1,2,3,4,5,6,7,8,9,10])
    train_Y = np.repeat(cls, 110)
    valid_Y = np.repeat(cls, 30)

    # feature normalizaiton
    train_X = train_X.T
    train_X_mean = np.mean(train_X, axis=0)
    train_X = train_X - train_X_mean
    train_X_std = np.std(train_X, axis=0)
    train_X = train_X / (train_X_std + 1e-5)
    
    valid_X = valid_X.T
    valid_X = valid_X - train_X_mean
    valid_X = valid_X/(train_X_std + 1e-5)

    # training model
    alphas = [1e-5, 5e-5, 0.0001, 0.0005, 0.001]

    model = []
    valid_acc = []
    for a in alphas:
        clf, acc = train_model(train_X, train_Y, valid_X, valid_Y, a, classifier='svc')
        model.append(clf)
        valid_acc.append(acc)
        
    # choose the model that achieve the best validation accuracy
    final_model = model[np.argmax(valid_acc)]

    # now, evaluate the model with the test set
    valid_Y_hat = final_model.predict(valid_X)

    accuracy = np.sum((valid_Y_hat == valid_Y))/300.0*100.0
    print('final validation accuracy = ' + str(accuracy) + ' %')

