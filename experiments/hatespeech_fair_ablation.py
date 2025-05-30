# import logging
# import os
# import numpy as np
# import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.set_loglevel('warning')  
# import torch
# import torch.optim as optim
# from datasetsdefer.hatespeech import HateSpeech
# from methods.FairCostCombined import PL_Combine_Fair
# from networks.linear_net import LinearNet
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
from methods.faircomb import *
from methods.runningEOD import *
from methods.FairCostCombined import PL_Combine_Fair_Cost

# from networks.cnn import *
# from networks.cnn import DenseNet121_CE, NetSimple, WideResNet

import fairlearn.metrics as metrics

# Configure logging and directories
logging.basicConfig(level=logging.INFO)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
data_dir = '../data'
# [previous imports remain the same...

def main():
    logging.info("using device:" + torch.cuda.get_device_name(device))
    # Ensure directories exist
    os.makedirs("../exp_data/plots", exist_ok=True)

    # Hyperparameters
    K_values = [1, 5, 10, 20, 30]
    threshold_range = np.arange(0.0, 1, 0.09)  # Adjust range as needed
    total_epochs = 100
    lr = 1e-2

    # Load dataset once for consistency
    dataset = HateSpeech(data_dir, True, False, 'random_annotator', device)

    results = {}

    for K in K_values:
        eqd_data = []

        # Initialize model and fairness method
        model = LinearNet(dataset.d, 4).to(device)
        pl_fair = PL_Combine_Fair(model, device, k=K)
        
        # Train the model
        pl_fair.fit(
            dataset.data_train_loader,
            dataset.data_val_loader,
            dataset.data_test_loader,
            epochs=total_epochs,
            optimizer=optim.Adam,
            lr=lr,
            verbose=False,
            test_interval=5
        )
        for r in threshold_range:
            # Evaluate
            output = pl_fair.test(dataset.data_test_loader, r)
            metrics = print_metrics(output, class_num=3, combine_method="PL")
            # Calculate deferral rate
            deferral_rate = np.mean(output['defers'])
            # Collect metrics
            eqd_data.append({
                'threshold': r,
                'EQD_c0': metrics.get('system_equalized_odds_difference_c0', 0),
                'EQD_c1': metrics.get('system_equalized_odds_difference_c1', 0),
                'EQD_c2': metrics.get('system_equalized_odds_difference_c2', 0),
                'deferral_rate': deferral_rate
            })
            logging.info(f'K={K}, r={r} done.')
        
        results[K] = pd.DataFrame(eqd_data)

    # Generate plots for EQD
    for K, df in results.items():
        plt.figure(figsize=(10, 6))
        plt.plot(df['threshold'], df['EQD_c0'], label='Class 0')
        plt.plot(df['threshold'], df['EQD_c1'], label='Class 1')
        plt.plot(df['threshold'], df['EQD_c2'], label='Class 2')
        plt.plot(df['threshold'], df['deferral_rate'], label='Deferral Rate')

        plt.title(f'EOD and deferral rate vs Threshold (K={K})')
        plt.xlabel('Threshold (r)')
        plt.ylabel('EOD/Deferral Rate')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'../exp_data/plots/eqd_k{K}.png')
        plt.close()

    # # Generate plots for deferral rate
    # for K, df in results.items():
    #     plt.figure(figsize=(10, 6))
    #     plt.plot(df['threshold'], df['deferral_rate'], label='Deferral Rate')
    #     plt.title(f'Deferral Rate vs Threshold (K={K})')
    #     plt.xlabel('Threshold (r)')
    #     plt.ylabel('Deferral Rate')
    #     plt.legend()
    #     plt.grid(True)
    #     plt.savefig(f'../exp_data/plots/deferral_rate_k{K}.png')
    #     plt.close()

    # Save results to CSV for further analysis
    for K, df in results.items():
        df.to_csv(f'../exp_data/plots/results_k{K}.csv', index=False)

def print_metrics(data, class_num=3, combine_method="PL"):
    # Reuse the existing print_metrics function to extract metrics
    res = {}
    for positive_class in range(class_num):
        combined_preds = np.array(data["combined_preds"]) == positive_class
        labels = data["labels"] == positive_class
        demographics = data["demographics"]
        
        # Calculate Equalized Odds Difference
        eod = metrics.equalized_odds_difference(
            labels, combined_preds, sensitive_features=demographics
        )
        res[f'system_equalized_odds_difference_c{positive_class}'] = eod
    return res

if __name__ == "__main__":
    main()