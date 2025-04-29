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

# from methods.combination_methods import UnsupervisedEMCombiner
# from methods.oraclecombiner2 import OracleCombinerDynamicEOD
from methods.oracleCombinerFairCost import OracleCombinerFairCost

eps_cst = 1e-8

def set_seed(seed: int):
    """Set the seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class PL_Combine_Fair_Cost(BaseMethod):
    """Selective Prediction method, train classifier on all data, and defer based on thresholding classifier confidence (max class prob)"""

    def __init__(self, model_class, device, plotting_interval=100, 
                k=20, r=0.05, lambda_parity=1.0, parity_threshold=0.1):
        self.model_class = model_class
        self.device = device
        self.plotting_interval = plotting_interval
        # Initialize combiner with fairness-cost parameters
        self.combiner = OracleCombinerFairCost(
            k=k,
            parity_threshold=parity_threshold,
            lambda_parity=lambda_parity,
            calibration_method='temperature scaling'
        )
        set_seed(42)
        self.features_train = []  

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
            # cross entropy loss
            loss = F.cross_entropy(outputs, data_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            prec1 = accuracy(outputs.data, data_y, topk=(1,))[0]
            losses.update(loss.data.item(), data_x.size(0))
            top1.update(prec1.item(), data_x.size(0))

            # measure elapsed time
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
        class_probs_all = []
        demographics_all = []
        features_all = []  # Store raw features for KNN

        self.model_class.eval()
        with torch.no_grad():
            for batch, (data_x, data_y, hum_preds, demographics) in enumerate(dataloader):
                data_x = data_x.to(self.device)
                outputs_class = self.model_class(data_x)
                outputs_class = F.softmax(outputs_class, dim=1)
                
                # Store features (flatten if needed)
                features_all.append(data_x.cpu().numpy().reshape(len(data_x), -1))  # [batch_size, features]
                
                # Rest of the collection
                max_class_probs, predicted_class = torch.max(outputs_class.data, 1)
                truths_all.extend(data_y.cpu().numpy())
                hum_preds_all.extend(hum_preds.cpu().numpy())
                class_probs_all.extend(outputs_class.cpu().numpy())
                demographics_all.extend(demographics.cpu().numpy())

        # Convert to numpy arrays
        features_all = np.concatenate(features_all, axis=0)
        class_probs_all = np.array(class_probs_all)
        truths_all = np.array(truths_all)
        hum_preds_all = np.array(hum_preds_all)
        demographics_all = np.array(demographics_all)

        # Fit combiner with features for KNN
        self.combiner.fit(
            model_probs=class_probs_all,
            y_h=hum_preds_all,
            y_true=truths_all,
            demographics=demographics_all,
            features=features_all
        )
    def fit(
        self,
        dataloader_train,
        dataloader_val,
        dataloader_test,
        epochs,
        optimizer,
        lr,
        verbose=True,
        test_interval=5,
        scheduler=None,
    ):
        # fit classifier and expert same time
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
        # change here to fit combiner
        self.fit_combiner(dataloader_train)

        return compute_deferral_metrics(self.test(dataloader_test))

    def test(self, dataloader):
        defers_all = []
        combined_preds_all = []
        
        self.model_class.eval()
        with torch.no_grad():
            for batch, (data_x, data_y, hum_preds, demographics) in enumerate(dataloader):
                data_x = data_x.to(self.device)
                outputs_class = self.model_class(data_x)
                outputs_class = F.softmax(outputs_class, dim=1)
                
                # Get test features
                test_features = data_x.cpu().numpy().reshape(len(data_x), -1)
                
                # Pass features to combine_proba
                combined_probs, defers, _ = self.combiner.combine_proba(
                    outputs_class.cpu().numpy(),
                    hum_preds.cpu().numpy(),
                    data_y.cpu().numpy(),
                    features_test=test_features
                )
                
                combined_preds, _ = self.combiner.combine(outputs_class.cpu().numpy(), hum_preds.cpu().numpy(), data_y.cpu().numpy())
                
                predictions_all.extend(predicted_class.cpu().numpy())
                truths_all.extend(data_y.cpu().numpy())
                hum_preds_all.extend(hum_preds.cpu().numpy())
                class_probs_all.extend(outputs_class.cpu().numpy())
                demographics_all.extend(demographics.cpu().numpy())
                max_probs.extend(max_class_probs.cpu().numpy())
                combined_probs_all.extend(combined_probs) # already a numpy array
                combined_preds_all.extend(combined_preds)

                # defers = np.ones(len(data_y)) # change this to take input from combiner
                # defers = np.array(defers)
                defers_all.extend(defers)
        # Convert to numpy
        defers_all = np.array(defers_all)
        truths_all = np.array(truths_all)
        hum_preds_all = np.array(hum_preds_all)
        predictions_all = np.array(predictions_all)
        max_probs = np.array(max_probs)
        rej_score_all = np.array(rej_score_all)
        class_probs_all = np.array(class_probs_all)
        demographics_all = np.array(demographics_all)
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
        }
        return data
