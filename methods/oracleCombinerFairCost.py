import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import time
# import torch.utils.data as data
import sys
import logging
from tqdm import tqdm
import random
import os

sys.path.append("..")
from helpers.utils import *
from helpers.metrics import *
from baselines.basemethod import BaseMethod

from sklearn.neighbors import NearestNeighbors
# from methods.oraclecombiner import OracleCombiner
from methods.oraclecombiner import OracleCombiner



class OracleCombinerFairCost(OracleCombiner):
    """OracleCombiner with KNN fairness: defers if (human_cost + λ * parity) < (P_X + threshold)"""
    
    def __init__(self, calibration_method='temperature scaling', k=20, parity_threshold=0.05, lambda_parity=1.0, **kwargs):
        super().__init__(calibration_method, **kwargs)
        self.k = k                   # Number of neighbors for KNN
        self.parity_threshold = parity_threshold  # Threshold for cost-fairness trade-off
        self.lambda_parity = lambda_parity        # Weight for parity term
        self.nn_dem0 = None          # KNN models for demographic groups
        self.nn_dem1 = None
        self.nn_dem0_labels = None   # True labels for neighbors
        self.nn_dem0_preds = None    # Model predictions for neighbors
        self.nn_dem1_labels = None
        self.nn_dem1_preds = None

    def fit(self, model_probs, y_h, y_true, demographics, features):
        """Fit KNN models for fairness and calibrator for cost."""
        # 1. Fit the base OracleCombiner (confusion matrix + calibration)
        super().fit(model_probs, y_h, y_true)
        
        # 2. Split features by demographic and fit KNN
        features_dem0, labels_dem0, preds_dem0 = [], [], []
        features_dem1, labels_dem1, preds_dem1 = [], [], []
        
        for i, demo in enumerate(demographics):
            if demo == 0:
                features_dem0.append(features[i])
                labels_dem0.append(y_true[i])
                preds_dem0.append(np.argmax(model_probs[i]))
            else:
                features_dem1.append(features[i])
                labels_dem1.append(y_true[i])
                preds_dem1.append(np.argmax(model_probs[i]))
        
        # Train KNN models
        self.nn_dem0 = NearestNeighbors(n_neighbors=self.k).fit(features_dem0)
        self.nn_dem1 = NearestNeighbors(n_neighbors=self.k).fit(features_dem1)
        self.nn_dem0_labels = np.array(labels_dem0)
        self.nn_dem0_preds = np.array(preds_dem0)
        self.nn_dem1_labels = np.array(labels_dem1)
        self.nn_dem1_preds = np.array(preds_dem1)

    def combine_proba(self, model_probs, y_h, y_true_te, features_test, miss_cost=9, human_cost=1):
        """Deferral condition: (human_cost + λ*parity) < (P_X + threshold)"""
        n_samples = model_probs.shape[0]
        calibrated_probs = self.calibrate(model_probs)
        model_preds = np.argmax(calibrated_probs, axis=1)
        defers = np.zeros(n_samples, dtype=int)
        y_comb = np.zeros((n_samples, self.n_cls))
        
        for i in range(n_samples):
            # Compute model's expected cost (P_X)
            P_X = miss_cost * (1 - calibrated_probs[i][model_preds[i]])
            
            # Compute fairness parity using KNN
            feat = features_test[i].reshape(1, -1)
            _, indices0 = self.nn_dem0.kneighbors(feat)
            _, indices1 = self.nn_dem1.kneighbors(feat)
            acc0 = np.mean(self.nn_dem0_preds[indices0[0]] == self.nn_dem0_labels[indices0[0]])
            acc1 = np.mean(self.nn_dem1_preds[indices1[0]] == self.nn_dem1_labels[indices1[0]])
            parity = abs(acc0 - acc1)  # Statistical disparity
            
            # Deferral condition
            if human_cost + self.lambda_parity * parity < P_X + self.parity_threshold:
                y_comb[i] = calibrated_probs[i] * self.confusion_matrix[y_h[i]]
                defers[i] = 1
            else:
                y_comb[i] = calibrated_probs[i]
                defers[i] = 0
        
        # Normalize and return
        y_comb /= np.sum(y_comb, axis=1, keepdims=True)
        return y_comb, defers, {}