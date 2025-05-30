import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import time
# import torch.utils.data as data
import sys
import logging
from tqdm import tqdm

sys.path.append("..")
from helpers.utils import *
from helpers.metrics import *
from baselines.basemethod import BaseMethod
import math

# from methods.combination_methods import UnsupervisedEMCombiner
from methods.allcombiner import AllCombiner

from sklearn.neighbors import NearestNeighbors
eps_cst = 1e-8

# def set_seed(seed: int):
#     """Set the seed for reproducibility."""
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed(seed)
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark = False
class PL_Combine_Fair(BaseMethod):
    """Selective Prediction method, train classifier on all data, and defer based on fairness cost and human cost"""

    def __init__(self, model_class, device, plotting_interval=100, 
                 k=None, fairness_cost=9.0, human_cost=1.0):
        super().__init__()
        self.model_class = model_class
        self.device = device
        self.plotting_interval = plotting_interval
        self.combiner = AllCombiner()
        self.k = k  # Number of neighbors for k-NN
        self.fairness_cost = fairness_cost  # Cost of unfairness (replaces self.r)
        self.human_cost = human_cost  # Cost of consulting a human
        self.nn_dem0 = None
        self.nn_dem1 = None

    def fit_epoch_class(self, dataloader, optimizer, verbose=True, epoch=1):
        batch_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        end = time.time()
        loss_fn = nn.CrossEntropyLoss()

        self.model_class.train()
        for batch, (data_x, data_y, hum_preds, demographics) in enumerate(dataloader):
            data_x = data_x.to(self.device)
            data_y = data_y.to(self.device)
            outputs = self.model_class(data_x)
            loss = F.cross_entropy(outputs, data_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            prec1 = accuracy(outputs.data, data_y, topk=(1,))[0]
            losses.update(loss.data.item(), data_x.size(0))
            top1.update(prec1.item(), data_x.size(0))

            batch_time.update(time.time() - end)
            end = time.time()
            if torch.isnan(loss):
                print("Nan loss")
                logging.warning(f"NAN LOSS")
                break
            if verbose and batch % self.plotting_interval == 0:
                logging.info(
                    "Epoch: [{0}][{1}/{2}]\t"
                    "Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                    "Loss {loss.val:.4f} ({loss.avg:.4f})\t"
                    "Prec@1 {top1.val:.3f} ({top1.avg:.3f})".format(
                        epoch,
                        batch,
                        len(dataloader),
                        batch_time=batch_time,
                        loss=losses,
                        top1=top1,
                    )
                )

    def fit_combiner(self, dataloader):
        max_probs = []
        truths_all = []
        hum_preds_all = []
        predictions_all = []   # classifier predictions
        class_probs_all = []   # classifier probabilities
        
        features_dem0, labels_dem0, preds_dem0 = [], [], []
        features_dem1, labels_dem1, preds_dem1 = [], [], []

        if self.k is None:
            num_samples = len(dataloader.dataset)
            self.k = max(5, min(30, int(math.log2(num_samples))))
        logging.info(f"value of K chosen = {self.k}")
        
        self.model_class.eval()
        with torch.no_grad():
            for batch, (data_x, data_y, hum_preds, demographics) in enumerate(dataloader):
                data_x = data_x.to(self.device)
                data_y = data_y.to(self.device)
                hum_preds = hum_preds.to(self.device)
                demographics = demographics.to(self.device)

                outputs_class = self.model_class(data_x)
                outputs_class = F.softmax(outputs_class, dim=1)
                max_class_probs, predicted_class = torch.max(outputs_class.data, 1)
                
                predictions_all.extend(predicted_class.cpu().numpy())
                truths_all.extend(data_y.cpu().numpy())
                hum_preds_all.extend(hum_preds.cpu().numpy())
                class_probs_all.extend(outputs_class.cpu().numpy())
                max_probs.extend(max_class_probs.cpu().numpy())
                
                for i in range(data_x.size(0)):
                    demo = demographics[i].item()
                    feature_vector = data_x[i].cpu().numpy()
                    pred = predicted_class[i].cpu().numpy()
                    if demo == 0:
                        features_dem0.append(feature_vector)
                        labels_dem0.append(data_y[i].cpu().numpy())
                        preds_dem0.append(pred)
                    elif demo == 1:
                        features_dem1.append(feature_vector)
                        labels_dem1.append(data_y[i].cpu().numpy())
                        preds_dem1.append(pred)
                        
        truths_all = np.array(truths_all)
        hum_preds_all = np.array(hum_preds_all)
        predictions_all = np.array(predictions_all)
        max_probs = np.array(max_probs)
        class_probs_all = np.array(class_probs_all)
        
        # Store training labels and predictions for disparity calculation
        self.y_train = truths_all
        self.y_pred_train = predictions_all
        
        self.combiner.fit(class_probs_all, hum_preds_all, truths_all)
        
        self.nn_dem0 = NearestNeighbors(n_neighbors=self.k).fit(features_dem0)
        self.nn_dem1 = NearestNeighbors(n_neighbors=self.k).fit(features_dem1)
        
        self.nn_dem0_labels = np.array(labels_dem0)
        self.nn_dem0_preds = np.array(preds_dem0)
        self.nn_dem1_labels = np.array(labels_dem1)
        self.nn_dem1_preds = np.array(preds_dem1)

    def test(self, dataloader, fairness_cost=None, human_cost=None):
        """
        Test the model and defer based on fairness cost and human cost.
        For each instance:
        - Compute local statistical parity using k-NN (absolute difference in accuracies).
        - Defer to human if human_cost <= fairness_cost * parity.
        - Track total cost (fairness cost for unfair predictions + human cost for deferrals).
        """
        if fairness_cost is not None:
            self.fairness_cost = fairness_cost
        if human_cost is not None:
            self.human_cost = human_cost

        defers_all = []
        max_probs = []
        truths_all = []
        hum_preds_all = []
        predictions_all = []   # classifier-only predictions
        rej_score_all = []     # rejector score (unused; set to 0)
        class_probs_all = []   # classifier probability vectors
        demographics_all = []
        combined_probs_all = []
        combined_preds_all = []
        
        # Track costs and correctness for metrics
        human_refered = 0
        human_correct = 0
        model_refered = 0
        model_correct = 0

        self.model_class.eval()
        with torch.no_grad():
            for batch, (data_x, data_y, hum_preds, demographics) in enumerate(dataloader):
                data_x = data_x.to(self.device)
                data_y = data_y.to(self.device)
                hum_preds = hum_preds.to(self.device)
                demographics = demographics.to(self.device)
                
                outputs_class = self.model_class(data_x)
                outputs_class = F.softmax(outputs_class, dim=1)
                max_class_probs, predicted_class = torch.max(outputs_class.data, 1)
                
                combined_probs, _ = self.combiner.combine_proba(
                    outputs_class.cpu().numpy(),
                    hum_preds.cpu().numpy(),
                    data_y.cpu().numpy()
                )
                combined_preds, _ = self.combiner.combine(
                    outputs_class.cpu().numpy(),
                    hum_preds.cpu().numpy(),
                    data_y.cpu().numpy()
                )
                
                outputs_np = outputs_class.cpu().numpy()
                predicted_class_np = predicted_class.cpu().numpy()
                
                batch_defers = []
                batch_combined_probs = []
                batch_combined_preds = []
                for i in range(len(data_y)):
                    point = data_x[i].cpu().numpy().reshape(1, -1)
                    
                    # Query k nearest neighbors for each demographic group
                    _, indices_dem0 = self.nn_dem0.kneighbors(point, n_neighbors=self.k)
                    _, indices_dem1 = self.nn_dem1.kneighbors(point, n_neighbors=self.k)
                    
                    # Retrieve training predictions and true labels
                    group0_preds = np.array(self.nn_dem0_preds)[indices_dem0[0]]
                    group0_labels = np.array(self.nn_dem0_labels)[indices_dem0[0]]
                    group1_preds = np.array(self.nn_dem1_preds)[indices_dem1[0]]
                    group1_labels = np.array(self.nn_dem1_labels)[indices_dem1[0]]
                    
                    # Compute local accuracy for each group
                    acc0 = np.mean(group0_preds == group0_labels) if len(group0_preds) > 0 else 0
                    acc1 = np.mean(group1_preds == group1_labels) if len(group1_preds) > 0 else 0
                    
                    # Compute statistical parity
                    parity = abs(acc0 - acc1)
                    
                    # # Compute fairness cost for deferral decision
                    # fairness_risk = self.fairness_cost * parity
                    
                    # Defer if human_cost <= fairness_risk
                    if self.human_cost <= self.fairness_cost * parity:
                        batch_defers.append(1)
                        batch_combined_probs.append(combined_probs[i])
                        batch_combined_preds.append(hum_preds[i].cpu().item())
                        human_refered += 1
                        if hum_preds[i].cpu().item() == data_y[i].cpu().item():
                            human_correct += 1
                    else:
                        batch_defers.append(0)
                        batch_combined_probs.append(outputs_np[i])
                        batch_combined_preds.append(predicted_class_np[i])
                        model_refered += 1
                        if predicted_class_np[i] == data_y[i].cpu().item():
                            model_correct += 1
                
                predictions_all.extend(predicted_class_np)
                truths_all.extend(data_y.cpu().numpy())
                hum_preds_all.extend(hum_preds.cpu().numpy())
                class_probs_all.extend(outputs_np)
                demographics_all.extend(demographics.cpu().numpy())
                max_probs.extend(max_class_probs.cpu().numpy())
                combined_probs_all.extend(batch_combined_probs)
                combined_preds_all.extend(batch_combined_preds)
                defers_all.extend(batch_defers)
                rej_score_all.extend(np.zeros(len(data_y)))
        
        defers_all = np.array(defers_all)
        truths_all = np.array(truths_all)
        hum_preds_all = np.array(hum_preds_all)
        predictions_all = np.array(predictions_all)
        max_probs = np.array(max_probs)
        rej_score_all = np.array(rej_score_all)
        class_probs_all = np.array(class_probs_all)
        demographics_all = np.array(demographics_all)
        combined_probs_all = np.array(combined_probs_all)
        combined_preds_all = np.array(combined_preds_all)
        
        # Compute total cost
        total_cost = self.fairness_cost * sum(abs(np.array(self.nn_dem0_preds)[indices_dem0[0]] != 
                                                 np.array(self.nn_dem0_labels)[indices_dem0[0]]) +
                                              abs(np.array(self.nn_dem1_preds)[indices_dem1[0]] != 
                                                  np.array(self.nn_dem1_labels)[indices_dem1[0]]) 
                                              for _ in range(model_refered)) / self.k + \
                     self.human_cost * human_refered
        
        data = {
            "defers": defers_all,
            "labels": truths_all,
            "max_probs": max_probs,
            "hum_preds": hum_preds_all,
            "preds": predictions_all,
            "rej_score": rej_score_all,
            "class_probs": class_probs_all,
            "demographics": demographics_all,
            "combined_probs": combined_probs_all,
            "combined_preds": combined_preds_all,
            # "fairness_cost": self.fairness_cost,
            # "human_cost": self.human_cost,
            # "total_cost": total_cost,
            # "human_refered": human_refered,
            # "human_correct": human_correct,
            # "human_accuracy": human_correct / human_refered if human_refered > 0 else 0,
            # "model_refered": model_refered,
            # "model_correct": model_correct,
            # "model_accuracy": model_correct / model_refered if model_refered > 0 else 0,
            # "combined_accuracy": (human_correct + model_correct) / len(truths_all)
        }
        return data

    def fit(self, dataloader_train, dataloader_val, dataloader_test, epochs, optimizer, lr, verbose=True, test_interval=5, scheduler=None):
        optimizer_class = optimizer(self.model_class.parameters(), lr=lr)
        if scheduler is not None:
            scheduler = scheduler(optimizer, len(dataloader_train)*epochs)

        self.model_class.train()
        for epoch in tqdm(range(epochs)):
            self.fit_epoch_class(
                dataloader_train, optimizer_class, verbose=verbose, epoch=epoch
            )
            if verbose and epoch % test_interval == 0:
                logging.info(compute_classification_metrics(self.test(dataloader_val)))
            if scheduler is not None:
                scheduler.step()
        
        self.fit_combiner(dataloader_train)
        # return compute_deferral_metrics(self.test(dataloader_test))
