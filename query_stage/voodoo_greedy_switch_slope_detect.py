#Greedy-switching algorithm
from time import time
import numpy as np
import math
from sklearn.cluster import KMeans
from voodoo_greedy_balance_slope_detect import voodoo_greedy_balance_slope_detect
import copy
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

class MnistResNet(ResNet):
    def __init__(self):
        super(MnistResNet, self).__init__(BasicBlock, [2, 2, 2, 2], num_classes=10)
        self.conv1 = torch.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        
    def forward(self, x):
        return torch.softmax(super(MnistResNet, self).forward(x), dim=-1)

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

def voodoo_greedy_switch_slope_detect(slope_list, num_test, n_cluster, clusters_pre, data, batch_size, n_trials, ucb_alpha, label_to_find):
    np.random.seed(0)
    time_voodoo = []
    results_voodoo = []
    n_item = len(clusters_pre)
    time_flag = []

    device = torch.device("cuda:2")
    model = MnistResNet()
    model.load_state_dict(torch.load("/z/wenjiah/MNIST_pytorch/MNIST_model"))
    model.to(device)
    model.eval()
    data_transform = Compose([ToTensor(), Normalize((torch.tensor(33.3100)/255,), (torch.tensor(78.5675)/255,))])

    for trial in range(n_trials):
        item_acc = []
        total_pulls = 0
        ori_index = np.arange(n_cluster)

        parent = []
        children = []
        n_arms = n_cluster
        temp = n_cluster
        while(int(temp/2) > 0):
            for i in range(temp):
                parent.append(int(n_arms+i/2))
            temp = math.ceil(temp/2)
            n_arms += temp
        parent.append(-1)
        start = 0
        for i in range(n_cluster):
            children.append([-1])
        temp = n_cluster
        while(int(temp/2) > 0):
            for i in range(math.ceil(temp/2)):
                if(2*i+1 < temp):
                    children.append([start+2*i, start+2*i+1])
                else:
                    children.append([start+2*i])
            start += temp
            temp = math.ceil(temp/2)

        clusters = []
        for i in range(n_cluster):
            clusters.append([])
        for i in range(len(clusters_pre)):
            clusters[clusters_pre[i]].append(i)
        for i in range(n_cluster):
            np.random.shuffle(clusters[i])
        
        found = np.zeros(n_arms)    #number of accessed relevant items in each cluster
        counts = np.zeros(n_arms)   #number of accessed items in each cluster
        ucbs = np.zeros(n_arms)     #upper confidence bound of each cluster
        ucb_mask = np.ones(n_arms)  #record which cluster is empty
        for i in range(n_arms):
            ucbs[i] = float("inf")
        ucbs[n_arms-1] = 1.
        label_found = np.ones(n_item)*(-1)

        init_time = time()
        
        clusters, found, counts, ucbs, ucb_mask, item_acc_balance, time_balance, ori_index, label_found = voodoo_greedy_balance_slope_detect(model, data_transform, device, slope_list[2], clusters, n_cluster, parent, children, found, counts, ucbs, ucb_mask, data, label_to_find, ucb_alpha, total_pulls, batch_size, label_found, not_sort = True, ori_dendro = False, ori_index = ori_index)
        time_voodoo += time_balance
        total_pulls += slope_list[2]
        item_acc += item_acc_balance
        time_flag = np.arange((batch_size-1),slope_list[2],batch_size)
        if(time_flag[-1] != slope_list[2]-1):
            time_flag = np.append(time_flag, slope_list[2]-1)

        r_cur = item_acc.count(label_to_find)/len(item_acc)
        if (r_cur < 0.1):
            n_scan = int(1.96**2*r_cur*(1-r_cur)/(r_cur/2)**2)
        else:
            n_scan = int(1.96**2*r_cur*(1-r_cur)/(r_cur/5)**2)
        print("the number of scanning:%d"%n_scan)
        r_scan = label_found[:n_scan][label_found[:n_scan]==label_to_find].size
        scanidx = 0
        while(scanidx < n_scan):
            if(scanidx+batch_size < n_scan):
                group_size = batch_size
            else:
                group_size = n_scan - scanidx
            feeditems = []
            feedindex = []
            for i in range(group_size):
                if(label_found[scanidx+i] == -1):
                    feedindex.append(scanidx+i)
                    feeditems.append(data[scanidx+i])
            if(len(feedindex) > 0):
                results = UDF(feeditems, model, data_transform, device)
                r_scan += results.count(label_to_find)
                label_found[feedindex] = results
            scanidx += group_size
        r_scan /= n_scan
        time_voodoo[-1] = time() - init_time
        if(r_cur < r_scan):
            cur = 0
            feeditems = []
            for i in range(n_cluster):
                while(len(clusters[i])>0):
                    item = clusters[i].pop()
                    if(label_found[item] == -1):
                        feeditems.append(data[item])
                    else:
                        item_acc.append(label_found[item])
                    cur += 1
                    if(cur == batch_size):
                        if(len(feeditems)>0):
                            results = UDF(feeditems, model, data_transform, device)
                            item_acc += results
                            feeditems = []
                        time_voodoo.append(time()-init_time)
                        cur = 0
            if(cur > 0):
                if(len(feeditems)>0):
                    results = UDF(feeditems, model, data_transform, device)
                    item_acc += results
                time_voodoo.append(time()-init_time)
            time_flag = np.append(time_flag, np.arange(time_flag[-1]+batch_size,len(clusters_pre),batch_size))
            if(time_flag[-1] != len(clusters_pre)-1):
                time_flag = np.append(time_flag, len(clusters_pre)-1)
        else:
            for i in range(2,len(slope_list)):
                lacc = item_acc[slope_list[i-1]:slope_list[i]].count(label_to_find)
                llacc = item_acc[slope_list[i-2]:slope_list[i-1]].count(label_to_find)
                if(lacc <= llacc*0.8):
                    curtime_sort = time()
                    clusters_sort, found_sort, counts_sort, ucbs_sort, ucb_mask_sort, item_acc_sort, time_sort, ori_index_sort, label_found = voodoo_greedy_balance_slope_detect(model, data_transform, device, num_test, copy.deepcopy(clusters), n_cluster, parent, children, copy.deepcopy(found), copy.deepcopy(counts), copy.deepcopy(ucbs), copy.deepcopy(ucb_mask), data, label_to_find, ucb_alpha, total_pulls, batch_size, label_found, not_sort=False, ori_dendro=False, ori_index=copy.deepcopy(ori_index))
                    curtime_unsort = time()
                    clusters_unsort, found_unsort, counts_unsort, ucbs_unsort, ucb_mask_unsort, item_acc_unsort, time_unsort, ori_index_unsort, label_found = voodoo_greedy_balance_slope_detect(model, data_transform, device, num_test, copy.deepcopy(clusters), n_cluster, parent, children, copy.deepcopy(found), copy.deepcopy(counts), copy.deepcopy(ucbs), copy.deepcopy(ucb_mask), data, label_to_find, ucb_alpha, total_pulls, batch_size, label_found, not_sort=True, ori_dendro=True, ori_index=copy.deepcopy(ori_index))
                    total_pulls += num_test
                    if(item_acc_unsort.count(label_to_find) > item_acc_sort.count(label_to_find)):
                        print("Original dendrogram")
                        time_voodoo += [item+curtime_unsort-init_time for item in time_unsort]
                        item_acc += item_acc_unsort
                        ori_index = ori_index_unsort
                        not_sort = True
                    else:
                        print("Sort leaf clusters")
                        time_voodoo += [item+curtime_sort-init_time for item in time_sort]
                        item_acc += item_acc_sort
                        ori_index = ori_index_sort
                        not_sort = False
                    time_flag = np.append(time_flag, np.arange(time_flag[-1]+batch_size,len(item_acc),batch_size))
                    if(time_flag[-1] != len(item_acc)-1):
                        time_flag = np.append(time_flag, len(item_acc)-1)
                    
                    if(i!=len(slope_list)-1):
                        next_target = slope_list[i+1]
                    else:
                        next_target = n_item
                    curtime = time()
                    if(not_sort):
                        clusters, found, counts, ucbs, ucb_mask, item_acc_balance, time_balance, ori_index, label_found = voodoo_greedy_balance_slope_detect(model, data_transform, device, next_target-total_pulls, clusters_unsort, n_cluster, parent, children, found_unsort, counts_unsort, ucbs_unsort, ucb_mask_unsort, data, label_to_find, ucb_alpha, total_pulls, batch_size, label_found, not_sort = True, ori_dendro = False, ori_index=ori_index)
                    else:
                        clusters, found, counts, ucbs, ucb_mask, item_acc_balance, time_balance, ori_index, label_found = voodoo_greedy_balance_slope_detect(model, data_transform, device, next_target-total_pulls, clusters_sort, n_cluster, parent, children, found_sort, counts_sort, ucbs_sort, ucb_mask_sort, data, label_to_find, ucb_alpha, total_pulls, batch_size, label_found, not_sort = True, ori_dendro = False, ori_index=ori_index)
                    time_voodoo += [item+curtime-init_time for item in time_balance]
                    total_pulls = next_target
                    item_acc += item_acc_balance
                    time_flag = np.append(time_flag, np.arange(time_flag[-1]+batch_size,len(item_acc),batch_size))
                    if(time_flag[-1] != len(item_acc)-1):
                        time_flag = np.append(time_flag, len(item_acc)-1)
                else:
                    if(i!=len(slope_list)-1):
                        next_target = slope_list[i+1]
                    else:
                        next_target = n_item
                    curtime = time()
                    clusters, found, counts, ucbs, ucb_mask, item_acc_balance, time_balance, ori_index, label_found = voodoo_greedy_balance_slope_detect(model, data_transform, device, next_target-total_pulls, clusters, n_cluster, parent, children, found, counts, ucbs, ucb_mask, data, label_to_find, ucb_alpha, total_pulls, batch_size, label_found, not_sort = True, ori_dendro = False, ori_index=ori_index)
                    time_voodoo += [item+curtime-init_time for item in time_balance]
                    total_pulls = next_target
                    item_acc += item_acc_balance
                    time_flag = np.append(time_flag, np.arange(time_flag[-1]+batch_size,len(item_acc),batch_size))
                    if(time_flag[-1] != len(item_acc)-1):
                        time_flag = np.append(time_flag, len(item_acc)-1)
        
        access = np.array(item_acc)
        access[access != label_to_find] = -1
        access[access == label_to_find] = 0
        access += 1
        results_voodoo.append(access)
    ave = np.average(results_voodoo, axis=0)
    results_voodoo.append(ave)

    time_voodoo = np.array(time_voodoo)

    torch.cuda.empty_cache()

    return results_voodoo, time_voodoo, time_flag