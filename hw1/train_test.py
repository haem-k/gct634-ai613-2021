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
import utils
from feature_summary import *

from sklearn.decomposition import PCA
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from scipy.fftpack import dct




def train_model(train_X, train_Y, valid_X, valid_Y, hyper_param1, options):

    # Choose a classifier (here, linear SVM)
    if options.classifier == 'sgd':
        clf = SGDClassifier(verbose=0, loss="hinge", alpha=hyper_param1, max_iter=1000, penalty="l2", random_state=0)
    elif options.classifier == 'knn':
        clf = KNeighborsClassifier(n_neighbors=options.neighbors)
    elif options.classifier == 'svc':
        clf = SVC()

    # train
    clf.fit(train_X, train_Y)

    # validation
    valid_Y_hat = clf.predict(valid_X)

    accuracy = np.sum((valid_Y_hat == valid_Y))/300.0*100.0
    if options.classifier == 'sgd':
        print(f'alpha = {hyper_param1}, validation accuracy = {accuracy}%')
    
    return clf, accuracy




if __name__ == '__main__':
    options = utils.train_parser()
    print(f"\nReceived options:\n{options}\n")



    '''
    Preprocess MFCC features
    '''
    if options.mfcc_delta == 'none':
        # load mfcc 
        train_X = load_mfcc('train')
        valid_X = load_mfcc('valid')

    elif options.mfcc_delta == 'd':
        # load mfcc delta
        train_X = delta_mfcc(dataset='train')
        valid_X = delta_mfcc(dataset='valid')

    elif options.mfcc_delta == 'dd':
        # load mfcc double delta
        train_X = double_delta_mfcc(dataset='train')
        valid_X = double_delta_mfcc(dataset='valid')
    
    train_X = np.array(train_X)
    valid_X = np.array(valid_X)

    # Reshape to make 1D vector for each audio file
    train_data_num = train_X.shape[0]
    valid_data_num = valid_X.shape[0]
    
    train_X = np.reshape(train_X, (train_data_num, -1))
    valid_X = np.reshape(valid_X, (valid_data_num, -1))



    '''
    Preprocess RMS features
    '''
    if options.rms_delta == 'none':
        # load rms 
        rms_train = load_rms('train')
        rms_valid = load_rms('valid')

    elif options.rms_delta == 'd':
        # load rms delta
        rms_train = delta_rms('train')
        rms_valid = delta_rms('valid')
        
    elif options.rms_delta == 'dd':
        # load rms double delta
        rms_train = double_delta_rms('train')
        rms_valid = double_delta_rms('valid')

    train_X = np.concatenate((train_X, rms_train), axis=1)
    valid_X = np.concatenate((valid_X, rms_valid), axis=1)



    '''
    Feature vector compression
    '''
    print(f"train_X shape before {options.compress}: {train_X.shape}")
    
    # Compress 
    if options.compress == 'pca':
        pca = PCA(n_components=options.dimension)
        train_X = pca.fit_transform(train_X)
        valid_X = pca.transform(valid_X)

    elif options.compress == 'dct':
        train_X = dct(train_X, n=options.dimension)
        valid_X = dct(valid_X, n=options.dimension)

    print(f"train_X shape after {options.compress}: {train_X.shape}")



    '''
    Prepare training
    '''
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
    train_X_mean = np.mean(train_X, axis=0)
    train_X = train_X - train_X_mean
    train_X_std = np.std(train_X, axis=0)
    train_X = train_X / (train_X_std + 1e-5)
    
    valid_X = valid_X - train_X_mean
    valid_X = valid_X/(train_X_std + 1e-5)

    # training model
    alphas = [1e-5, 5e-5, 0.0001, 0.0005, 0.001]

    model = []
    valid_acc = []
    for a in alphas:
        clf, acc = train_model(train_X, train_Y, valid_X, valid_Y, a, options)
        model.append(clf)
        valid_acc.append(acc)
        
    # choose the model that achieve the best validation accuracy
    final_model = model[np.argmax(valid_acc)]

    # now, evaluate the model with the test set
    valid_Y_hat = final_model.predict(valid_X)

    accuracy = np.sum((valid_Y_hat == valid_Y))/300.0*100.0
    print('Final validation accuracy = ' + str(accuracy) + ' %')

    # Keep track of wrong guesses
    wrong = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    
    # Print all wrong guesses and correct answers
    print()
    for i in range(0, 300):
        if valid_Y_hat[i] != valid_Y[i]:
            print(f'Index: {i}, Answer: {valid_Y[i]}, Wrong guess: {valid_Y_hat[i]}')
            wrong[valid_Y[i]-1] += 1
    print(f'Wrong guesses: {wrong}')

        

