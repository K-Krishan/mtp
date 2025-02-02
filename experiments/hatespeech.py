import logging
import os
import pickle
import sys

import torch
import torch.optim as optim

sys.path.append("../")
import sys

import torch
import torch.nn as nn

sys.path.append("../")
import argparse
import datetime
# allow logging to print everything
import logging

# from baselines.lce_surrogate import *
# from datasetsdefer.synthetic_data import SyntheticData
# from helpers.metrics import *
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

from methods.costcombination import *
from methods.combination import *

# from networks.cnn import *
# from networks.cnn import DenseNet121_CE, NetSimple, WideResNet

import fairlearn.metrics as metrics

def combine_defer(preds, h_preds, defers):
    return preds * (1 - defers) + h_preds * defers
def print_metrics(data, class_num=3, combine_method="defer"):
    res = dict()
    for positive_class in range(class_num):  # Loop over each class
        preds = (data["preds"] == positive_class)
        labels = (data["labels"] == positive_class)
        hpreds = (data["hum_preds"] == positive_class)
        defers = data["defers"]
        demographics = data["demographics"]
        # print(data["combined_preds"])
        
        if combine_method == "defer":
            combined_preds = combine_defer(preds, hpreds, defers)
        elif combine_method == "PL":
            combined_preds = np.array(data["combined_preds"])
            combined_preds = (combined_preds == positive_class)

        # Metrics for each class and demographic group
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

        # Global metrics for demographic parity and equalized odds
        res[f"model_demographic_parity_difference_c{positive_class}"] = metrics.demographic_parity_difference(labels, preds, sensitive_features=demographics)
        res[f"human_demographic_parity_difference_c{positive_class}"] = metrics.demographic_parity_difference(labels, hpreds, sensitive_features=demographics)
        res[f"system_demographic_parity_c{positive_class}"] = metrics.demographic_parity_difference(labels, combined_preds, sensitive_features=demographics)

        res[f"model_equalized_odds_difference_c{positive_class}"] = metrics.equalized_odds_difference(labels, preds, sensitive_features=demographics)
        res[f"human_equalized_odds_difference_c{positive_class}"] = metrics.equalized_odds_difference(labels, hpreds, sensitive_features=demographics)
        res[f"system_equalized_odds_difference_c{positive_class}"] = metrics.equalized_odds_difference(labels, combined_preds, sensitive_features=demographics)

    # General performance metrics
    res['deferral rate'] = data["defers"].mean()
    res['model accuracy'] = (data['preds'] == data["labels"]).mean()
    res['human accuracy'] = (data["hum_preds"] == data["labels"]).mean()
    if combine_method == "defer":
        combined_preds = combine_defer(preds, hpreds, defers)
    elif combine_method == "PL":
        combined_preds = np.array(data["combined_preds"])
    res['combined accuracy'] = (combined_preds == data["labels"]).mean()

    # Print results
    for k, v in res.items():
        print(k, ":", v)

    return res


def summarize_metrics(trial_results):
    """
    Summarize metrics across trials by calculating the mean and variance for each metric.

    Parameters:
    trial_results (list of dict): A list of dictionaries, where each dictionary contains metrics from a trial.

    Returns:
    None: Prints the mean and variance for each metric.
    """
    if not trial_results:
        print("No trial results to summarize.")
        return

    # Extract all keys from the first trial's dictionary
    keys = trial_results[0].keys()

    # Compute mean and variance for each metric
    summary = {}
    for key in keys:
        values = [trial[key] for trial in trial_results if key in trial]
        mean = np.mean(values)
        variance = np.var(values)
        summary[key] = {"mean": mean, "variance": variance}

    # Print the summary
    for key, stats in summary.items():
        print(f"{key}: {stats['mean']}")
        # print(f"  Mean: {stats['mean']}")
        # print(f"  Variance: {stats['variance']}")

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
    
    data_dir = '../data'
    

    optimizer = optim.Adam
    scheduler = None
    lr = 1e-2
    max_trials = 5 # 5 
    total_epochs = 100 # 100

    # consider changing to NonLinearNet
    stats_combination_cost = []
    stats_combination_all = []
    for trial in range(max_trials):

        # generate data
        dataset = HateSpeech(data_dir, True, False, 'random_annotator', device)
        # dataset = HateSpeech(data_dir, True, False, 'synthetic', device, synth_exp_param=[0.95, 0.92])

        # combination with cost
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
        stats_combination_cost.append(print_metrics(output, class_num=3, combine_method="PL"))

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
        stats_combination_all.append(print_metrics(output, class_num=3, combine_method="PL"))

    print('\n\n--Stats of cost optimized unsupervised P+L combiation')
    summarize_metrics(stats_combination_cost)
    print('\n\n--Stats of all combiation')
    summarize_metrics(stats_combination_all)
    


if __name__ == "__main__":
    main()
