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

from baselines.lce_surrogate import *
from datasetsdefer.synthetic_data import SyntheticData
from helpers.metrics import *
from networks.linear_net import *

logging.basicConfig(level=logging.DEBUG)
import datetime

import torch.optim as optim
from baselines.compare_confidence import *
from baselines.differentiable_triage import *
from baselines.lce_surrogate import *
from baselines.mix_of_exps import *
from baselines.one_v_all import *
from baselines.selective_prediction import *
from datasetsdefer.broward import *
from datasetsdefer.chestxray import *
from datasetsdefer.cifar_h import *
from datasetsdefer.cifar_synth import *
from datasetsdefer.generic_dataset import *
from datasetsdefer.hatespeech import *
from datasetsdefer.imagenet_16h import *
from datasetsdefer.synthetic_data import *
from methods.milpdefer import *
from methods.realizable_surrogate import *
from networks.cnn import *
from networks.cnn import DenseNet121_CE, NetSimple, WideResNet




device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

data_dir = '../data'
dataset = HateSpeech(data_dir, True, False, 'random_annotator', device)

optimizer = optim.Adam
scheduler = None
lr = 1e-2
max_trials = 10# 5
total_epochs = 100 # 100

print("Device name:", torch.cuda.get_device_name(torch.cuda.current_device()))
# Load model components
model_class = LinearNet(dataset.d, 3).to(device)
model_class.load_state_dict(torch.load("../exp_data/models/model_class.pth"))

model_rejector = LinearNet(dataset.d, 2).to(device)
model_rejector.load_state_dict(torch.load("../exp_data/models/model_rejector.pth"))

# Load DifferentiableTriage-specific parameters
diff_triage = DifferentiableTriage(model_class, model_rejector, device, 0.000, "human_error")
print(diff_triage.test(dataset.data_test_loader))

# def test(self, dataloader):
#     defers_all = []
#     truths_all = []
#     hum_preds_all = []
#     predictions_all = []  # classifier only
#     rej_score_all = []  # rejector probability
#     class_probs_all = []  # classifier probability
#     self.model_rejector.eval()
#     self.model_class.eval()
#     with torch.no_grad():
#         for batch, (data_x, data_y, hum_preds) in enumerate(dataloader):
#             data_x = data_x.to(self.device)
#             data_y = data_y.to(self.device)
#             hum_preds = hum_preds.to(self.device)
#             outputs_class = self.model_class(data_x)
#             outputs_class = F.softmax(outputs_class, dim=1)
#             outputs_rejector = self.model_rejector(data_x)
#             outputs_rejector = F.softmax(outputs_rejector, dim=1)
#             _, predictions_rejector = torch.max(outputs_rejector.data, 1)
#             max_class_probs, predicted_class = torch.max(outputs_class.data, 1)
#             predictions_all.extend(predicted_class.cpu().numpy())
#             truths_all.extend(data_y.cpu().numpy())
#             hum_preds_all.extend(hum_preds.cpu().numpy())
#             defers_all.extend(predictions_rejector.cpu().numpy())
#             rej_score_all.extend(outputs_rejector[:, 1].cpu().numpy())
#             class_probs_all.extend(outputs_class.cpu().numpy())
#     # convert to numpy
#     defers_all = np.array(defers_all)
#     truths_all = np.array(truths_all)
#     hum_preds_all = np.array(hum_preds_all)
#     predictions_all = np.array(predictions_all)
#     rej_score_all = np.array(rej_score_all)
#     class_probs_all = np.array(class_probs_all)
#     data = {
#         "defers": defers_all,
#         "labels": truths_all,
#         "hum_preds": hum_preds_all,
#         "preds": predictions_all,
#         "rej_score": rej_score_all,
#         "class_probs": class_probs_all,
#     }
#     return data