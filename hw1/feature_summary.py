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

MFCC_DIM = 13

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



def delta_mfcc(dataset='train'):
    f = open(data_path + dataset + '_list.txt', 'r')
    total_delta = []
    
    i = 0
    for file_name in f:
        # load mfcc file
        file_name = file_name.rstrip('\n')
        file_name = file_name.replace('.wav','.npy')
        mfcc_file = mfcc_path + file_name
        mfcc = np.load(mfcc_file)
        mfcc = mfcc.transpose()

        # compute delta
        frames = mfcc.shape[0]
        delta_mat = []
        for j in range(1, frames):
            delta = mfcc[j] - mfcc[j-1]
            delta_mat.append(delta)
        delta_mat = np.array(delta_mat)
        total_delta.append(delta_mat)
        i = i + 1

    f.close()

    total_delta = np.array(total_delta)
    print(total_delta.shape)
    return total_delta




def double_delta_mfcc(dataset='train'):
    f = open(data_path + dataset + '_list.txt', 'r')
    double_delta_mean_mat = []
    
    i = 0
    for file_name in f:
        # load mfcc file
        file_name = file_name.rstrip('\n')
        file_name = file_name.replace('.wav','.npy')
        mfcc_file = mfcc_path + file_name
        mfcc = np.load(mfcc_file)
        mfcc = mfcc.transpose()

        # compute delta
        frames = mfcc.shape[0]
        delta_mat = []
        for j in range(1, frames):
            delta = mfcc[j] - mfcc[j-1]
            delta_mat.append(delta)
        delta_mat = np.array(delta_mat)

        # compute double delta
        num_delta = delta_mat.shape[0]
        double_delta_mat = []
        for j in range(1, num_delta):
            double_delta = delta_mat[j] - delta_mat[j-1]
            double_delta_mat.append(double_delta)
        double_delta_mat = np.array(double_delta_mat)

        # mean pooling double delta
        double_delta_mean = np.mean(double_delta_mat, axis=0)
        double_delta_mean_mat.append(double_delta_mean)

        i = i + 1

    f.close()
    
    double_delta_mean_mat = np.array(double_delta_mean_mat).transpose()
    return double_delta_mean_mat


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

    rms_mat = np.array(rms_mat)
    return rms_mat



def delta_rms(dataset='train'):
    f = open(data_path + dataset + '_list.txt', 'r')
    rms_mat = []
    rms_delta_mat = []

    i = 0
    for file_name in f:
        # load rms file
        file_name = file_name.rstrip('\n')
        file_name = file_name.replace('.wav','.npy')
        rms_file = rms_path + file_name
        rms = np.load(rms_file)
        rms_mat.append(rms)
        
        # compute delta
        rms_dim = rms.shape[0]
        delta = []
        for j in range(1, rms_dim):
            delta.append(rms[j] - rms[j-1])
        delta = np.array(delta)
        rms_delta_mat.append(delta)

        i = i + 1

    f.close()

    rms_delta_mat = np.array(rms_delta_mat)
    print(rms_delta_mat.shape)
    return rms_delta_mat





if __name__ == '__main__':
    delta_mfcc()
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








