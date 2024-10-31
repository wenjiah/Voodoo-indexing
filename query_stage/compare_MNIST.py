from time import time
import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.io import loadmat
from scipy.spatial import distance

from sklearn.cluster import KMeans
from sklearn.datasets import load_digits, fetch_mldata
from sklearn.preprocessing import scale

from voodoo_greedy_switch_slope_detect import voodoo_greedy_switch_slope_detect

raw_data = loadmat("/z/wenjiah/mnist-original.mat")

#shuffle the last n_samps data to cluster
np.random.seed(0)
n_samps = 70000
samp_idx = np.arange(n_samps)
np.random.shuffle(samp_idx)
data_ori = np.transpose(raw_data["data"])
label_ori = raw_data["label"][0]
data = data_ori[samp_idx,:]
label = label_ori[samp_idx]

n_cluster = 1000
clusters_pre = np.load("mnist_k1000.npy").astype("int")

#set parameters
batch_size = 1
n_trials = 1
ucb_alpha = 1.
results = {}
label_to_find_list = [0,1,2,3,4,5,6,7,8,9] 
num_test = int(n_samps/100)
slope_list = (np.arange(0,1,0.05)*n_samps).astype("int")

for n in range(10):
        label_to_find = label_to_find_list[n]
        results["Voodoo greedy switch detect"], time_voodoogd, time_flag= voodoo_greedy_switch_slope_detect(slope_list, num_test, n_cluster, clusters_pre, data, batch_size, n_trials, ucb_alpha, label_to_find)