from time import time
import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.io import loadmat
from scipy.spatial import distance

from sklearn.cluster import KMeans
from sklearn.datasets import load_digits, fetch_mldata
from sklearn.preprocessing import scale

raw_data = loadmat("/z/wenjiah/mnist-original.mat")

#shuffle the last n_samps data to cluster
np.random.seed(0)
n_samps = 70000
samp_idx = np.arange(n_samps)
np.random.shuffle(samp_idx)
data_ori = np.transpose(raw_data["data"])
label_ori = raw_data["label"][0]
data = scale(data_ori[samp_idx,:].astype(np.float64))
label = label_ori[samp_idx]

#use K-means to do clustering
kmeans = KMeans(init = 'k-means++', n_clusters = 1000, n_init = 10)
kmeans.fit(data)
clusters_pre = kmeans.predict(data)

n_cluster = 1000
cluster_seq = []
for i in range(n_cluster):
    cluster_seq.append([])
    cluster_seq[i].append(i)
cluster_center_ = kmeans.cluster_centers_
cluster_center = []
for i in range(len(cluster_center_)):
    cluster_center.append(cluster_center_[i])
while(len(cluster_seq) > 1):
    cluster_seq_new = []
    cluster_center_new = []
    dis = distance.pdist(cluster_center)
    while(len(cluster_center) > 1):
        min_dis = min(dis)
        dis = distance.squareform(dis)
        index1, index2 = np.where(dis == min_dis)[0]
        cluster_seq_new.append(cluster_seq[index1]+cluster_seq[index2])
        cluster_center_new.append((cluster_center[index1]+cluster_center[index2])/2)
        cluster_seq.pop(index2)
        cluster_seq.pop(index1)
        cluster_center.pop(index2)
        cluster_center.pop(index1)
        dis = np.delete(dis, [index1, index2], axis=0)
        dis = np.delete(dis, [index1, index2], axis=1)
        dis = distance.squareform(dis)
    if(len(cluster_center) == 1):
        cluster_seq_new.append(cluster_seq[0])
        cluster_center_new.append(cluster_center[0])
    cluster_seq = cluster_seq_new
    cluster_center = cluster_center_new
cluster_seq = cluster_seq[0]
true_index = []
for i in range(n_cluster):
    true_index.append(cluster_seq.index(i))
for i in range(len(clusters_pre)):
    clusters_pre[i] = true_index[clusters_pre[i]]

np.save("mnist_k1000.npy", clusters_pre)
