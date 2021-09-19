import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from feature_summary import *


data_path = './dataset/'
mfcc_path = './mfcc/'
rms_path = './rms/'

MFCC_DIM = 13

