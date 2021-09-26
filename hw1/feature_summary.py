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



'''
MFCC related functions
'''
def load_mfcc(dataset='train'):
    
    f = open(data_path + dataset + '_list.txt','r')
    mfcc_mat = []

    i = 0
    for file_name in f:

        # load mfcc file
        file_name = file_name.rstrip('\n')
        file_name = file_name.replace('.wav','.npy')
        mfcc_file = mfcc_path + file_name
        mfcc = np.load(mfcc_file)
        mfcc = mfcc.transpose()
        mfcc_mat.append(mfcc)
        i = i + 1

    f.close()

    mfcc_mat = np.array(mfcc_mat)
    print(f'{dataset} MFCC shape: {mfcc_mat.shape}')
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
    print(f'{dataset} MFCC delta shape: {total_delta.shape}')
    return total_delta

def double_delta_mfcc(dataset='train'):
    f = open(data_path + dataset + '_list.txt', 'r')
    total_double_delta = []
    
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
        total_double_delta.append(double_delta_mat)
        i = i + 1

    f.close()
    
    total_double_delta = np.array(total_double_delta)
    print(f'{dataset} MFCC double delta shape: {total_double_delta.shape}')
    return total_double_delta

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



'''
RMS related functions
'''
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
    print(f'{dataset} RMS shape: {rms_mat.shape}')
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
        rms = np.load(rms_file)     # (173,)
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
    print(f'{dataset} RMS delta shape: {rms_delta_mat.shape}')
    return rms_delta_mat



def double_delta_rms(dataset='train'):
    f = open(data_path + dataset + '_list.txt', 'r')
    total_double_delta = []

    i = 0
    for file_name in f:
        # load rms file
        file_name = file_name.rstrip('\n')
        file_name = file_name.replace('.wav','.npy')
        rms_file = rms_path + file_name
        rms = np.load(rms_file)     # (173,)
        
        # compute delta
        rms_dim = rms.shape[0]
        delta = []
        for j in range(1, rms_dim):
            delta.append(rms[j] - rms[j-1])
        delta = np.array(delta)

        # compute double delta
        num_delta = delta.shape[0]
        double_delta_mat = []
        for j in range(1, num_delta):
            double_delta = delta[j] - delta[j-1]
            double_delta_mat.append(double_delta)

        double_delta_mat = np.array(double_delta_mat)
        total_double_delta.append(double_delta_mat)

        i = i + 1

    f.close()

    total_double_delta = np.array(total_double_delta)
    print(f'{dataset} RMS double delta shape: {total_double_delta.shape}')
    return total_double_delta






if __name__ == '__main__':
    load_mfcc()
    delta_mfcc()
    double_delta_mfcc()

    print()

    load_rms()
    delta_rms()
    double_delta_rms()

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








