# GCT634 (2018) HW1 
#
# Mar-18-2018: initial version
# 
# Juhan Nam
#

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from numpy.core.fromnumeric import shape

data_path = './dataset/'
mfcc_path = './mfcc/'
rms_path = './rms/'

MFCC_DIM = 20

def mean_mfcc(dataset='train'):
    
    f = open(data_path + dataset + '_list.txt','r')

    if dataset == 'train':
        mfcc_mat = np.zeros(shape=(MFCC_DIM, 1100))
    else:
        mfcc_mat = np.zeros(shape=(MFCC_DIM, 300))

    i = 0
    for file_name in f:

        # load mfcc file
        file_name = file_name.rstrip('\n')
        file_name = file_name.replace('.wav','.npy')
        mfcc_file = mfcc_path + file_name
        mfcc = np.load(mfcc_file)

        # mean pooling
        temp = np.mean(mfcc, axis=1)
        mfcc_mat[:,i]= np.mean(mfcc, axis=1)
        i = i + 1

    f.close()

    return mfcc_mat



def load_rms(dataset='train'):
    f = open(data_path + dataset + '_list.txt','r')
    rms_mat = []

    i = 0
    for file_name in f:
        # load rms file
        file_name = file_name.rstrip('\n')
        file_name = file_name.replace('.wav','.npy')
        rms_file = rms_path + file_name
        rms = np.load(rms_file)

        # rms_mat.append()
        rms_mat.append(rms)
        i = i + 1

    f.close()

    rms_mat = np.array(rms_mat).transpose()
    print(rms_mat.shape)
    return rms_mat


def delta_feature(dataset='train', feature='rms'):
    f = open(data_path + dataset + '_list.txt', 'r')
    feature_mat = []
    feature_delta_mat = []

    if feature == 'rms':
        feature_path = rms_path
    elif feature == 'mfcc':
        feature_path = mfcc_path

    i = 0
    for file_name in f:
        # load feature file
        file_name = file_name.rstrip('\n')
        file_name = file_name.replace('.wav','.npy')
        feature_file = feature_path + file_name
        feature = np.load(feature_file)
        feature_mat.append(feature)
        
        # compute delta
        feature_dim = feature.shape[0]
        delta = []
        for j in range(1, feature_dim):
            delta.append(feature[j] - feature[j-1])
        delta = np.array(delta)
        feature_delta_mat.append(delta)

        i = i + 1

    f.close()

    feature_delta_mat = np.array(feature_delta_mat).transpose()
    return feature_delta_mat




if __name__ == '__main__':
    delta_feature()
    quit()
    train_data = mean_mfcc('train')
    valid_data = mean_mfcc('valid')

    plt.figure(1)
    plt.subplot(2,1,1)
    plt.imshow(train_data, interpolation='nearest', origin='lower', aspect='auto')
    plt.colorbar(format='%+2.0f dB')

    plt.subplot(2,1,2)
    plt.imshow(valid_data, interpolation='nearest', origin='lower', aspect='auto')
    plt.colorbar(format='%+2.0f dB')

    plt.show()








