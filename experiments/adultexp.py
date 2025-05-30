import logging
import os
import sys

sys.path.append("../")
import torch
import datetime
import numpy as np
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

def combine_defer(preds, h_preds, defers):
    return preds * (1 - defers) + h_preds * defers

def print_metrics(data, class_num=2, combine_method="defer"):
    res = dict()
    for positive_class in range(class_num):  # Loop over each class
        preds = (data["preds"] == positive_class)
        labels = (data["labels"] == positive_class)
        hpreds = (data["hum_preds"] == positive_class)
        defers = data["defers"]
        demographics = data["demographics"]
        
        if combine_method == "defer":
            combined_preds = combine_defer(preds, hpreds, defers)
        elif combine_method == "PL":
            combined_preds = np.array(data["combined_preds"])
            combined_preds = (combined_preds == positive_class)

        for demographic in set(demographics):
            demographic_mask = (demographics == demographic)
            demographic_labels = labels[demographic_mask]
            demographic_preds = preds[demographic_mask]
            demographic_combined_preds = combined_preds[demographic_mask]
            
            tp = ((demographic_combined_preds == 1) & (demographic_labels == 1)).sum()
            tn = ((demographic_combined_preds == 0) & (demographic_labels == 0)).sum()
            fp = ((demographic_combined_preds == 1) & (demographic_labels == 0)).sum()
            fn = ((demographic_combined_preds == 0) & (demographic_labels == 1)).sum()

            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
            tnr = tn / (tn + fp) if (tn + fp) > 0 else 0
            fnr = fn / (fn + tp) if (fn + tp) > 0 else 0

            res[f"Class {positive_class} | Demographic {demographic} | TPR"] = tpr
            res[f"Class {positive_class} | Demographic {demographic} | FPR"] = fpr
            res[f"Class {positive_class} | Demographic {demographic} | TNR"] = tnr
            res[f"Class {positive_class} | Demographic {demographic} | FNR"] = fnr

        res[f"model_demographic_parity_difference_c{positive_class}"] = metrics.demographic_parity_difference(labels, preds, sensitive_features=demographics)
        res[f"human_demographic_parity_difference_c{positive_class}"] = metrics.demographic_parity_difference(labels, hpreds, sensitive_features=demographics)
        res[f"system_demographic_parity_c{positive_class}"] = metrics.demographic_parity_difference(labels, combined_preds, sensitive_features=demographics)

        res[f"model_equalized_odds_difference_c{positive_class}"] = metrics.equalized_odds_difference(labels, preds, sensitive_features=demographics)
        res[f"human_equalized_odds_difference_c{positive_class}"] = metrics.equalized_odds_difference(labels, hpreds, sensitive_features=demographics)
        res[f"system_equalized_odds_difference_c{positive_class}"] = metrics.equalized_odds_difference(labels, combined_preds, sensitive_features=demographics)

    res['deferral rate'] = data["defers"].mean()
    res['model accuracy'] = (data['preds'] == data["labels"]).mean()
    res['human accuracy'] = (data["hum_preds"] == data["labels"]).mean()
    if combine_method == "defer":
        combined_preds = combine_defer(preds, hpreds, defers)
    elif combine_method == "PL":
        combined_preds = np.array(data["combined_preds"])
    res['combined accuracy'] = (combined_preds == data["labels"]).mean()

    for k, v in res.items():
        print(k, ":", v)

    return res

def summarize_metrics(trial_results):
    if not trial_results:
        print("No trial results to summarize.")
        return

    keys = trial_results[0].keys()
    summary = {}
    for key in keys:
        values = [trial[key] for trial in trial_results if key in trial]
        mean = np.mean(values)
        variance = np.var(values)
        summary[key] = {"mean": mean, "variance": variance}

    for key, stats in summary.items():
        print(f"{key}: {stats['mean']}")

def main():
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
    
    max_trials = 10
    total_epochs = 100

    stats_combination_cost = []
    stats_combination_all = []
    stats_combination_fair = []
    for trial in range(max_trials):

        dataset = Adult(data_dir, device)

        # model = GradientBoostingClassifier()
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
        print('\n\nFairness Metrics for cost optimized combination: ')
        stats_combination_cost.append(print_metrics(output, class_num=3, combine_method="PL"))

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
        print('\n\nFairness Metrics for all combination: ')
        stats_combination_all.append(print_metrics(output, class_num=3, combine_method="PL"))

        model = LinearNet(dataset.d,4).to(device)
        PLC = PL_Combine_Fair(model, device, fairness_cost=20.0, human_cost=1.0)
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
        # store_test_results_to_csv(output, csv_path='plc_test.csv')
        # sp_metrics = compute_coverage_v_acc_curve( output)
        print('\n\nFairness Metrics for Fair combination: ')
        stats_combination_fair.append(print_metrics(output, class_num=3, combine_method="PL"))
        

    print('\n\n--Stats of cost optimized unsupervised P+L combination')
    summarize_metrics(stats_combination_cost)
    print('\n\n--Stats of all combination')
    summarize_metrics(stats_combination_all)
    print('\n\n--Stats of fair combiation')
    summarize_metrics(stats_combination_fair)

if __name__ == "__main__":
    main()
