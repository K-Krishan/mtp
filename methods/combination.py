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

# from methods.combination_methods import UnsupervisedEMCombiner
from methods.allcombiner import AllCombiner

eps_cst = 1e-8

# def set_seed(seed: int):
#     """Set the seed for reproducibility."""
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed(seed)
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark = False


class PL_Combine(BaseMethod):
    """Selective Prediction method, train classifier on all data, and defer based on thresholding classifier confidence (max class prob)"""

    def __init__(self, model_class, device, plotting_interval=100):
        self.model_class = model_class
        self.device = device
        self.plotting_interval = plotting_interval
        self.combiner = AllCombiner()
        # set_seed(42)

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
        predictions_all = []  # classifier only
        # rej_score_all = []  # rejector probability
        class_probs_all = []  # classifier probability
        # model_probs = []
        # demographics_all = []
        self.model_class.eval()
        with torch.no_grad():
            for batch, (data_x, data_y, hum_preds, demographics) in enumerate(dataloader):
                data_x = data_x.to(self.device)
                data_y = data_y.to(self.device)
                hum_preds = hum_preds.to(self.device)
                # demographics = demographics.to(self.device)  # Ensure demographics are on the correct device
                outputs_class = self.model_class(data_x)
                outputs_class = F.softmax(outputs_class, dim=1)
                max_class_probs, predicted_class = torch.max(outputs_class.data, 1)
                
                # model_probs.extend(outputs_class.cpu().numpy())
                predictions_all.extend(predicted_class.cpu().numpy())
                truths_all.extend(data_y.cpu().numpy())
                hum_preds_all.extend(hum_preds.cpu().numpy())
                class_probs_all.extend(outputs_class.cpu().numpy())
                # demographics_all.extend(demographics.cpu().numpy())
                max_probs.extend(max_class_probs.cpu().numpy())
        # defers_all = np.array(defers_all)
        truths_all = np.array(truths_all)
        hum_preds_all = np.array(hum_preds_all)
        predictions_all = np.array(predictions_all)
        max_probs = np.array(max_probs) # model probabilities
        # rej_score_all = np.array(rej_score_all)
        class_probs_all = np.array(class_probs_all)
        # model_probs = np.array(model_probs)
        # demographics_all = np.array(demographics_all)

        # fitting the combiner here
        # print("max probs shape ",max_probs.shape)
        # print("predictions_all shape ", predictions_all.shape)
        print("human preds shape ",hum_preds_all.shape)
        print("class probs shape: ", class_probs_all.shape)
        
        self.combiner.fit(class_probs_all, hum_preds_all, truths_all)

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
        max_probs = []
        truths_all = []
        hum_preds_all = []
        predictions_all = []  # classifier only
        rej_score_all = []  # rejector probability
        class_probs_all = []  # classifier probability
        demographics_all = []
        combined_probs_all = []
        combined_preds_all = []
        self.model_class.eval()
        with torch.no_grad():
            for batch, (data_x, data_y, hum_preds, demographics) in enumerate(dataloader):
                data_x = data_x.to(self.device)
                data_y = data_y.to(self.device)
                hum_preds = hum_preds.to(self.device)
                
                demographics = demographics.to(self.device)  # Ensure demographics are on the correct device
                outputs_class = self.model_class(data_x)
                outputs_class = F.softmax(outputs_class, dim=1)
                max_class_probs, predicted_class = torch.max(outputs_class.data, 1)
                # print(outputs_class.shape, hum_preds.shape)
                combined_probs = self.combiner.combine_proba(outputs_class.cpu().numpy(), hum_preds.cpu().numpy(), data_y.cpu().numpy())
                combined_preds, _ = self.combiner.combine(outputs_class.cpu().numpy(), hum_preds.cpu().numpy(), data_y.cpu().numpy())
                
                predictions_all.extend(predicted_class.cpu().numpy())
                truths_all.extend(data_y.cpu().numpy())
                hum_preds_all.extend(hum_preds.cpu().numpy())
                class_probs_all.extend(outputs_class.cpu().numpy())
                demographics_all.extend(demographics.cpu().numpy())
                max_probs.extend(max_class_probs.cpu().numpy())
                combined_probs_all.extend(combined_probs) # already a numpy array
                combined_preds_all.extend(combined_preds)

                defers = np.ones(len(data_y))
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
