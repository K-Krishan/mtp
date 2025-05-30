import logging
import os
import sys

sys.path.append("../")
import torch
import datetime
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score

from datasetsdefer.AdultDataset import *
from methods.costcombination import *
from methods.combination import *
from networks.linear_net import *

import torch.optim as optim
import fairlearn.metrics as metrics

logging.basicConfig(level=logging.DEBUG)
from networks.linear_net import *

logging.basicConfig(level=logging.DEBUG)
import datetime


import torch.optim as optim
# from baselines.compare_confidence import *
# from baselines.differentiable_triage import *
# from baselines.lce_surrogate import *
# from baselines.mix_of_exps import *
# from baselines.one_v_all import *
# from baselines.selective_prediction import *
# from datasetsdefer.broward import *
# from datasetsdefer.chestxray import *
# from datasetsdefer.cifar_h import *
# from datasetsdefer.cifar_synth import *
# from datasetsdefer.generic_dataset import *
from datasetsdefer.hatespeech import *
# from datasetsdefer.imagenet_16h import *
# from datasetsdefer.synthetic_data import *
# from methods.milpdefer import *
# from methods.realizable_surrogate import *
from methods.seperate_thresholds import *

from methods.costcombination import PL_Combine_Cost
from methods.combination import PL_Combine
from methods.faircomb import PL_Combine_Fair
# from methods.runningEOD import *
# from methods.FairCostCombined import PL_Combine_Fair_Cost
import pandas as pd

if not os.path.exists("../exp_data"):
    os.makedirs("../exp_data")
    os.makedirs("../exp_data/data")
    os.makedirs("../exp_data/plots")
    os.makedirs("../exp_data/models")
else:
    if not os.path.exists("../exp_data/data"):
        os.makedirs("../exp_data/data")
    if not os.path.exists("../exp_data/plots"):
        os.makedirs("../exp_data/plots")
    if not os.path.exists("../exp_data/models"):    
        os.makedirs("../exp_data/models")

date_now = datetime.datetime.now()
date_now = date_now.strftime("%Y-%m-%d_%H%M%S")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
optimizer = optim.Adam
scheduler = None
lr = 1e-2

data_dir = '../data'

max_trials = 5
total_epochs = 100

dataset = Adult(data_dir, device)
# Initialize model
model = LinearNet(dataset.d,4).to(device)
PLC = PL_Combine_Fair(model, device, k=10, r=0.05)
PLC.fit(
    dataset.data_train_loader,
    dataset.data_val_loader,
    dataset.data_test_loader,
    epochs=total_epochs,
    optimizer=optimizer,
    scheduler=scheduler,
    lr=lr,
    verbose=False,
    test_interval=5,
)

# Get test results
test_results = PLC.test(dataset.data_test_loader)

# Extract predictions and labels
true_labels = test_results['labels']
human_preds = test_results['hum_preds']
model_preds = test_results['preds']
combined_preds = test_results['combined_preds']

# Function to plot confusion matrix
def plot_confusion_matrix(cm, title):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['0', '1', '2'], 
                yticklabels=['0', '1', '2'])
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

# Compute and plot confusion matrices
cm_human = confusion_matrix(true_labels, human_preds)
plot_confusion_matrix(cm_human, 'Human Confusion Matrix')

cm_model = confusion_matrix(true_labels, model_preds)
plot_confusion_matrix(cm_model, 'Model Confusion Matrix')

cm_combined = confusion_matrix(true_labels, combined_preds)
plot_confusion_matrix(cm_combined, 'Combined System Confusion Matrix')