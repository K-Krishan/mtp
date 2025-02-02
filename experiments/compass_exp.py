import logging
import os
import torch
import torch.optim as optim

import sys

import torch
sys.path.append("../")
import datetime
import logging

from networks.linear_net import *

logging.basicConfig(level=logging.DEBUG)
import datetime

import torch.optim as optim
from baselines.compare_confidence import *
from baselines.differentiable_triage import *
from baselines.selective_prediction import *
from datasetsdefer.broward import *
from hatespeech import *

def main():

    # check if there exists directory ../exp_data
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
    # {'classifier_all_acc': 0.9211, 'human_all_acc': 0.367, 'coverage': 0.9904, 'classifier_nondeferred_acc': 0.9262924071082391, 'human_deferred_acc': 0.3541666666666667, 'system_acc': 0.9208}
    
    data_dir = '../data'
    

    optimizer = optim.AdamW
    scheduler = None
    lr = 0.01
    max_trials = 5
    total_epochs = 500# 100
    stats_combination_cost = []
    stats_combination_all = []
    for trial in range(max_trials):

        # generate data
        dataset = BrowardDataset(data_dir, test_split = 0.2, val_split = 0.1)

        model = LinearNet(dataset.d,4).to(device)
        PLC = PL_Combine_Cost(model, device)
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
        output = PLC.test(dataset.data_test_loader)
        # sp_metrics = compute_coverage_v_acc_curve( output)
        print('\n\nFairness Metrics for cost optimized combination: ')
        stats_combination_cost.append(print_metrics(output, class_num=2, combine_method="PL"))

        # combination
        model = LinearNet(dataset.d,4).to(device)
        PLC = PL_Combine(model, device)
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
        output = PLC.test(dataset.data_test_loader)
        # sp_metrics = compute_coverage_v_acc_curve( output)
        print('\n\nFairness Metrics for all combination: ')
        stats_combination_all.append(print_metrics(output, class_num=2, combine_method="PL"))
    print('\n\n--Stats of cost optimized unsupervised P+L combiation')
    summarize_metrics(stats_combination_cost)
    print('\n\n--Stats of all combiation')
    summarize_metrics(stats_combination_all)
    

if __name__ == "__main__":
    main()
