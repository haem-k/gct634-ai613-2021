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

from sklearn.decomposition import PCA
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.mixture import GaussianMixture
from scipy.fftpack import dct




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

    train_data_num = train_dmfcc.shape[0]
    valid_data_num = valid_dmfcc.shape[0]
    
    train_dmfcc = np.reshape(train_dmfcc, (train_data_num, -1))
    valid_dmfcc = np.reshape(valid_dmfcc, (valid_data_num, -1))
    print(f"train_dmfcc shape before: {train_dmfcc.shape}")

    # PCA to mfcc delta
    # pca = PCA(n_components=0.90, svd_solver='full')
    pca = PCA(n_components=70)
    train_pca_dmfcc = pca.fit_transform(train_dmfcc)
    valid_pca_dmfcc = pca.transform(valid_dmfcc)
    print(f"train_dmfcc shape after pca: {train_pca_dmfcc.shape}")

    # # DCT to mfcc delta
    # dct_train_dmfcc = dct(train_dmfcc, n=50)
    # dct_valid_dmfcc = dct(valid_dmfcc, n=50)
    # print(f"train_rms shape after DCT: {dct_train_dmfcc.shape}")




    # # load mfcc double delta
    # train_dd = double_delta_mfcc(dataset='train')
    # valid_dd = double_delta_mfcc(dataset='valid')

    # load rms
    train_rms = load_rms('train')
    valid_rms = load_rms('valid')
    print(f"train_rms shape before : {train_rms.shape}")


    # # DCT to compress rms
    # dct_train_rms = dct(train_rms, n=20)
    # dct_valid_rms = dct(valid_rms, n=20)
    # print(f"train_rms shape after DCT: {dct_train_rms.shape}")


    # # PCA to compress rms
    # rms_pca = PCA(n_components=0.99, svd_solver='full')
    # pca_train_rms = rms_pca.fit_transform(train_rms)
    # pca_valid_rms = rms_pca.transform(valid_rms)
    # print(f"train_rms shape after PCA : {pca_train_rms.shape}")


    # # load rms delta
    # train_rms_delta = delta_rms('train')
    # valid_rms_delta = delta_rms('valid')

    # concat features
    train_X = np.concatenate((train_pca_dmfcc, train_rms), axis=1)
    valid_X = np.concatenate((valid_pca_dmfcc, valid_rms), axis=1)

    # train_X = np.concatenate((dct_train_dmfcc, dct_train_rms), axis=1)
    # valid_X = np.concatenate((dct_valid_dmfcc, dct_valid_rms), axis=1)

    # train_X = np.concatenate((train_pca_dmfcc, train_rms_delta), axis=1)
    # valid_X = np.concatenate((valid_pca_dmfcc, valid_rms_delta), axis=1)
    
    # train_X = np.concatenate((train_pca_dmfcc, pca_train_rms), axis=1)
    # valid_X = np.concatenate((valid_pca_dmfcc, pca_valid_rms), axis=1)
    
    # train_X = train_pca_dmfcc
    # valid_X = valid_pca_dmfcc

    print(f"-->  Input feature dimension: {train_X.shape}\n")

    # label generation
    # 1: bass_electronic
    # 2: brass_acoustic
    # 3: flute_acoustic
    # 4: guitar_acoustic
    # 5: keyboard_acoustic
    # 6: mallet_acoustic
    # 7: organ_electronic
    # 8: reed_acoustic
    # 9: string_acoustic
    # 10: vocal_acoustic
    cls = np.array([1,2,3,4,5,6,7,8,9,10])
    train_Y = np.repeat(cls, 110)
    valid_Y = np.repeat(cls, 30)

    # feature normalizaiton
    # train_X = train_X.T
    train_X_mean = np.mean(train_X, axis=0)
    train_X = train_X - train_X_mean
    train_X_std = np.std(train_X, axis=0)
    train_X = train_X / (train_X_std + 1e-5)
    
    # valid_X = valid_X.T
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
    wrong = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    
    for i in range(0, 300):
        if valid_Y_hat[i] != valid_Y[i]:
            print(f'Answer: {valid_Y[i]}, Wrong guess: {valid_Y_hat[i]}')
            wrong[valid_Y[i]-1] += 1
        
    print(f'wrong answers: {wrong}')
    print('final validation accuracy = ' + str(accuracy) + ' %')

