# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import scipy
import numpy as np
import torch.nn.functional as F
import torch
import scipy.stats
import time
import faiss
from faiss import normalize_L2
import os
import logging
logger = logging.getLogger()


def normalize(vecs):
    return vecs / ((vecs**2).sum(axis=1, keepdims=True)**0.5)

class WordKNN():
    def __init__(self, word_reprs, word_labels, normalize):
        self.word_reprs = word_reprs.detach().cpu().numpy()
        self.word_labels = word_labels.detach().cpu()
        self.normalize = normalize
        d = self.word_reprs.shape[1]
        res = faiss.StandardGpuResources()
        flat_config = faiss.GpuIndexFlatConfig()
        flat_config.device = torch.cuda.current_device()
        
        if self.normalize:
            self.index = faiss.GpuIndexFlatIP(res, d, flat_config)
            normalize_L2(self.word_reprs)
        self.index.add(self.word_reprs)
        return
    

    def search_knn(self, qy_reprs, k):
        qy_reprs = qy_reprs.detach().cpu().numpy()
        if self.normalize:
            normalize_L2(qy_reprs)
        
        # D: distance matrix, (qy_num, k + 10), I: index matrix (qy_num, k + 10)
        D, I = self.index.search(qy_reprs, k * 2)
        # remove extractly match instances and keep k
        D = torch.from_numpy(D).type(torch.float32)
        I = torch.from_numpy(I).type(torch.long)
        same_flag = (D < 1 + 1e-7) & (D > 1 - 1e-7) 

        same_nums = same_flag.sum(-1)
        knn_dists = []
        knn_inds = []
        knn_labels = []
        for i in range(qy_reprs.shape[0]):
            knn_dists.append(D[i, same_nums[i]: same_nums[i] + k])
            knn_inds.append(I[i, same_nums[i]: same_nums[i] + k])
            knn_labels.append(self.word_labels[knn_inds[-1]])
        knn_dists = torch.stack(knn_dists, dim=0)
        knn_inds = torch.stack(knn_inds, dim=0)
        knn_lbs = torch.stack(knn_labels, dim=0)
        return knn_dists, knn_inds, knn_lbs