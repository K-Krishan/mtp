import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import time
import sys
import logging
from tqdm import tqdm
import sklearn.metrics

sys.path.append("..")
from helpers.utils import AverageMeter, accuracy
from baselines.basemethod import BaseMethod
eps_cst = 1e-8

class SelectivePredictionSep(BaseMethod):
    """Selective Prediction method, train classifier on all data, and defer based on thresholding classifier confidence (max class prob)"""

    def __init__(self, model_class, device, plotting_interval=100):
        self.model_class = model_class
        self.device = device
        self.plotting_interval = plotting_interval
        self.demographic0 = 0.5
        self.demographic1 = 0.5
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
            # if verbose and batch % self.plotting_interval == 0:
            #     logging.info(
            #         "Epoch: [{0}][{1}/{2}]\t"
            #         "Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
            #         "Loss {loss.val:.4f} ({loss.avg:.4f})\t"
            #         "Prec@1 {top1.val:.3f} ({top1.avg:.3f})".format(
            #             epoch,
            #             batch,
            #             len(dataloader),
            #             batch_time=batch_time,
            #             loss=losses,
            #             top1=top1,
            #         )
            #     )

    def set_optimal_threshold(self, dataloader):
        """Set separate thresholds to maximize system accuracy on validation set based on demographic key.
        Args:
            dataloader (_type_): dataloader validation set
        """
        data_preds = self.test(dataloader)
        treshold_grid = data_preds["max_probs"]
        treshold_grid = np.append(treshold_grid, np.linspace(0, 1, 20))

        best_treshold_0 = 0
        best_acc_0 = 0
        best_treshold_1 = 0
        best_acc_1 = 0

        # Separate predictions based on demographic key
        demographic_mask = data_preds["demographics"]  # True/False mask for demographic
        preds_demographic0 = {
            key: values[~demographic_mask] for key, values in data_preds.items()
        }
        preds_demographic1 = {
            key: values[demographic_mask] for key, values in data_preds.items()
        }

        # Optimize threshold for demographic == 0
        for treshold in treshold_grid:
            defers = (preds_demographic0["max_probs"] < treshold) * 1
            acc = sklearn.metrics.accuracy_score(
                preds_demographic0["preds"] * (1 - defers) +
                preds_demographic0["hum_preds"] * (defers),
                preds_demographic0["labels"],
            )
            if acc > best_acc_0:
                best_acc_0 = acc
                best_treshold_0 = treshold

        # Optimize threshold for demographic == 1
        for treshold in treshold_grid:
            defers = (preds_demographic1["max_probs"] < treshold) * 1
            acc = sklearn.metrics.accuracy_score(
                preds_demographic1["preds"] * (1 - defers) +
                preds_demographic1["hum_preds"] * (defers),
                preds_demographic1["labels"],
            )
            if acc > best_acc_1:
                best_acc_1 = acc
                best_treshold_1 = treshold

        logging.info(
            f"Best threshold for demographic 0: {best_treshold_0} with accuracy {best_acc_0}"
        )
        logging.info(
            f"Best threshold for demographic 1: {best_treshold_1} with accuracy {best_acc_1}"
        )

        self.demographic0 = best_treshold_0
        self.demographic1 = best_treshold_1


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
            # if verbose and epoch % test_interval == 0:
            #     logging.info(compute_classification_metrics(self.test(dataloader_val)))
            if scheduler is not None:
                scheduler.step()
        self.set_optimal_threshold(dataloader_val)

        # return compute_deferral_metrics(self.test(dataloader_test))

    def test(self, dataloader):
        defers_all = []
        max_probs = []
        truths_all = []
        hum_preds_all = []
        predictions_all = []  # classifier only
        rej_score_all = []  # rejector probability
        class_probs_all = []  # classifier probability
        demographics_all = []
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
                predictions_all.extend(predicted_class.cpu().numpy())
                truths_all.extend(data_y.cpu().numpy())
                hum_preds_all.extend(hum_preds.cpu().numpy())
                class_probs_all.extend(outputs_class.cpu().numpy())
                demographics_all.extend(demographics.cpu().numpy())
                defers = []
                max_probs.extend(max_class_probs.cpu().numpy())
                for i in range(len(data_y)):
                    # Use demographic-based thresholds
                    if demographics[i].item() == 0:  # Demographic 0
                        threshold = self.demographic0
                    else:  # Demographic 1
                        threshold = self.demographic1
                    rej_score_all.extend([1 - max_class_probs[i].item()])
                    if max_class_probs[i] < threshold:
                        defers.extend([1])
                    else:
                        defers.extend([0])
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
        }
        return data
