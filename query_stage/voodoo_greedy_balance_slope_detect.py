#Balanced tree that will be switched to, used in voodoo_greedy_switch
from time import time
import numpy as np
import math
from sklearn.cluster import KMeans
import copy
import matplotlib.pyplot as plt
from scipy.stats import rankdata, kendalltau

from torchvision.models.resnet import ResNet, BasicBlock
from torchvision.datasets import MNIST
from tqdm.autonotebook import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import inspect
from torch import nn, optim
import torch
from torchvision.transforms import Compose, ToTensor, Normalize, Resize, ToPILImage
from torch.utils.data import DataLoader
import skimage
from torch.optim import lr_scheduler
from torchvision import datasets, models, transforms
import os
from PIL import Image
import flair

def UDF(feeditems, model, transform, device):
    num_row = len(feeditems)
    feeditems = np.array(feeditems).reshape([num_row,28,28])
    feedtran = torch.zeros([num_row,1,224,224])
    for i in range(num_row):
        feedtran[i] = transform(skimage.transform.resize(feeditems[i],(224,224)))
    feedtran = feedtran.to(device)
    result = model(feedtran)
    result = torch.argmax(result,dim=1).tolist()
    return result

def voodoo_greedy_balance_slope_detect(model, data_transform, device, num_pulls, clusters, n_cluster, parent, children, found, counts, ucbs, ucb_mask, data, label_to_find, ucb_alpha, total_pulls, batch_size, label_found, not_sort, ori_dendro, ori_index):
    item_acc = []
    time_balance = []

    time1 = time()

    if(not_sort == False):
        each_quality = found[:n_cluster]/counts[:n_cluster]
        sortindex = np.argsort(each_quality)
        sortclusters = []
        sortcounts = []
        sortfound = []
        sortmask = []
        for i in range(n_cluster):
            sortclusters.append(clusters[sortindex[n_cluster-i-1]])
            sortcounts.append(counts[sortindex[n_cluster-i-1]])
            sortfound.append(found[sortindex[n_cluster-i-1]])
            sortmask.append(ucb_mask[sortindex[n_cluster-i-1]])
        clusters = sortclusters
        counts[:n_cluster] = sortcounts
        found[:n_cluster] = sortfound
        ucb_mask[:n_cluster] = sortmask
        count_or_not = copy.deepcopy(ucb_mask)
        count_or_not[count_or_not == -1] = 0
        for i in range(n_cluster,len(counts)):
            l_child = children[i][0]
            if(len(children[i]) > 1):
                r_child = children[i][1]
            else:
                r_child = children[i][0]
            counts[i] = sum(counts[l_child:r_child+1]*count_or_not[l_child:r_child+1])
            found[i] = sum(found[l_child:r_child+1]*count_or_not[l_child:r_child+1])
            if(ucb_mask[l_child] == -1 and ucb_mask[r_child] == -1):
                ucb_mask[i] = -1
                count_or_not[i] = 0
            else:
                ucb_mask[i] = 1
                count_or_not[i] = 1
        sortindex = sortindex[::-1]
        newindex = np.zeros(n_cluster)
        for i in range(n_cluster):
            newindex[i] = ori_index[sortindex[i]]
        ori_index = newindex

    if(ori_dendro == True):
        oriclusters = copy.deepcopy(clusters)
        oricounts = copy.deepcopy(counts[:n_cluster])
        orifound = copy.deepcopy(found[:n_cluster])
        orimask = copy.deepcopy(ucb_mask[:n_cluster])
        for i in range(n_cluster):
            oriclusters[int(ori_index[i])] = clusters[i]
            oricounts[int(ori_index[i])] = counts[i]
            orifound[int(ori_index[i])] = found[i]
            orimask[int(ori_index[i])] = ucb_mask[i]
        clusters = oriclusters
        counts[:n_cluster] = oricounts
        found[:n_cluster] = orifound
        ucb_mask[:n_cluster] = orimask
        count_or_not = copy.deepcopy(ucb_mask)
        count_or_not[count_or_not == -1] = 0
        for i in range(n_cluster,len(counts)):
            l_child = children[i][0]
            if(len(children[i]) > 1):
                r_child = children[i][1]
            else:
                r_child = children[i][0]
            counts[i] = sum(counts[l_child:r_child+1]*count_or_not[l_child:r_child+1])
            found[i] = sum(found[l_child:r_child+1]*count_or_not[l_child:r_child+1])
            if(ucb_mask[l_child] == -1 and ucb_mask[r_child] == -1):
                ucb_mask[i] = -1
                count_or_not[i] = 0
            else:
                ucb_mask[i] = 1
                count_or_not[i] = 1
        ori_index = np.arange(n_cluster)
    
    for i in range(len(ucbs)):
        ucbs[i] = float("inf")
    ucbs[-1] = 1.

    pull = 0
    while(pull < num_pulls):
        if(num_pulls-pull > batch_size):
            group_size = batch_size
        else:
            group_size = num_pulls-pull

        ucbs = (found+1)/(counts+1) + ucb_alpha * np.sqrt(2. * np.log(total_pulls+1) / counts)
        ucbs *= ucb_mask

        cur_size = 0
        feedindex = []
        feeditems = []
        access_index = []
        access_num = []
        have_label = []
        found_label = []
        while(cur_size < group_size):
            #search from the root to find the next arm to pull
            next_pull = len(counts) - 1
            while(next_pull >= n_cluster and ucbs[next_pull] != float("inf")):
                if(len(children[next_pull]) == 1):
                    next_pull = children[next_pull][0]
                else:
                    if(ucbs[children[next_pull][0]] >= ucbs[children[next_pull][1]]):
                        next_pull = children[next_pull][0]
                    else:
                        next_pull = children[next_pull][1]
            if(next_pull < n_cluster):
                to_pull = next_pull
            else:
                l_child = children[next_pull][0]
                if(len(children[next_pull]) > 1):
                    r_child = children[next_pull][1]
                else:
                    r_child = children[next_pull][0]
                while(l_child >= n_cluster):
                    l_child = children[l_child][0]
                    if(len(children[r_child]) > 1):
                        r_child = children[r_child][1]
                    else:
                        r_child = children[r_child][0]
                to_pull = l_child + int(np.random.rand() * (r_child - l_child + 1))
                if(len(clusters[to_pull]) == 0):
                    to_pull = l_child  
                    while(len(clusters[to_pull]) == 0):
                        to_pull += 1
            if(len(clusters[to_pull])+cur_size <= group_size):
                cluster_size = len(clusters[to_pull])
                ucb_mask[to_pull] = -1.
                ucbs[to_pull] = -1
                idx = to_pull
                while(parent[idx] != -1):
                    idx = parent[idx]
                    if(len(children[idx]) == 1 and ucb_mask[children[idx][0]] == -1):
                        ucb_mask[idx] = -1
                        ucbs[idx] = -1
                    if(len(children[idx]) == 2 and ucb_mask[children[idx][0]] == -1 and ucb_mask[children[idx][1]] == -1):
                        ucb_mask[idx] = -1
                        ucbs[idx] = -1
            else:
                cluster_size = group_size-cur_size 

            access_num_cur = 0
            found_label_cur = 0
            for i in range(cluster_size):
                item = clusters[to_pull].pop()
                if(label_found[item] == -1):
                    feedindex.append(item)
                    feeditems.append(data[item])
                    access_num_cur += 1
                else:
                    item_acc.append(label_found[item])
                    if(label_found[item] == label_to_find):
                        found_label_cur += 1
            access_index.append(to_pull)
            access_num.append(access_num_cur)
            have_label.append(cluster_size-access_num_cur)
            found_label.append(found_label_cur)
            cur_size += cluster_size
        
        if(len(feeditems) > 0):
            results = UDF(feeditems, model, data_transform, device)
            item_acc += results
            for i in range(len(feedindex)):
                label_found[feedindex[i]] = results[i]
            for i in range(len(access_index)):
                countadd = access_num[i]+have_label[i]
                foundadd = results[:access_num[i]].count(label_to_find)+found_label[i]
                if(i != len(access_index)-1):
                    results = results[access_num[i]:]
                counts[access_index[i]] += countadd
                found[access_index[i]] += foundadd
                idx = access_index[i]
                while(parent[idx] != -1):
                    idx = parent[idx]
                    counts[idx] += countadd
                    found[idx] += foundadd
                    if(ucb_mask[access_index[i]] == -1 and counts[idx]-counts[access_index[i]]!=0):
                        counts[idx] -= counts[access_index[i]]
                        found[idx] -= found[access_index[i]]
        else:
            for i in range(len(access_index)):
                countadd = have_label[i]
                foundadd = found_label[i]
                counts[access_index[i]] += countadd
                found[access_index[i]] += foundadd
                idx = access_index[i]
                while(parent[idx] != -1):
                    idx = parent[idx]
                    counts[idx] += countadd
                    found[idx] += foundadd
                    if(ucb_mask[access_index[i]] == -1 and counts[idx]-counts[access_index[i]]!=0):
                        counts[idx] -= counts[access_index[i]]
                        found[idx] -= found[access_index[i]]
        time_balance.append(time()-time1)
        pull += group_size
        total_pulls += group_size

    return clusters, found, counts, ucbs, ucb_mask, item_acc, time_balance, ori_index, label_found
        