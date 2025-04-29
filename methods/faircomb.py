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
    """Selective Prediction method, train classifier on all data, and defer based on thresholding classifier confidence (max class prob)"""

    def __init__(self, model_class, device, plotting_interval=100, 
        k=20, r=0.05):
        super().__init__()
        self.model_class = model_class
        self.device = device
        self.plotting_interval = plotting_interval
        self.combiner = AllCombiner()
        self.k = k  # Number of neighbors for k-NN
        self.r = r  # Threshold for statistical disparity
        self.one_knn = None
        self.zero_knn = None


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
        # Containers for combiner training data
        max_probs = []
        truths_all = []
        hum_preds_all = []
        predictions_all = []   # classifier predictions
        class_probs_all = []   # classifier probabilities
        
        # Containers for nearest neighbor extraction per demographic group
        features_dem0, labels_dem0, preds_dem0 = [], [], []
        features_dem1, labels_dem1, preds_dem1 = [], [], []
        
        self.model_class.eval()
        with torch.no_grad():
            for batch, (data_x, data_y, hum_preds, demographics) in enumerate(dataloader):
                data_x = data_x.to(self.device)
                data_y = data_y.to(self.device)
                hum_preds = hum_preds.to(self.device)
                demographics = demographics.to(self.device)  # ensure demographics are on the correct device

                outputs_class = self.model_class(data_x)
                outputs_class = F.softmax(outputs_class, dim=1)
                max_class_probs, predicted_class = torch.max(outputs_class.data, 1)
                
                predictions_all.extend(predicted_class.cpu().numpy())
                truths_all.extend(data_y.cpu().numpy())
                hum_preds_all.extend(hum_preds.cpu().numpy())
                class_probs_all.extend(outputs_class.cpu().numpy())
                max_probs.extend(max_class_probs.cpu().numpy())
                
                # For each sample, extract and split by demographic
                for i in range(data_x.size(0)):
                    demo = demographics[i].item()  # 0 or 1
                    # Choose feature representation: raw input or a hidden layer output
                    feature_vector = data_x[i].cpu().numpy()  
                    # You can also choose to save predicted_class or outputs_class[i] depending on your use case
                    pred = predicted_class[i].cpu().numpy()
                    if demo == 0:
                        features_dem0.append(feature_vector)
                        labels_dem0.append(data_y[i].cpu().numpy())
                        preds_dem0.append(pred)
                    elif demo == 1:
                        features_dem1.append(feature_vector)
                        labels_dem1.append(data_y[i].cpu().numpy())
                        preds_dem1.append(pred)
                        
        # Convert lists to numpy arrays for combiner fitting
        truths_all = np.array(truths_all)
        hum_preds_all = np.array(hum_preds_all)
        predictions_all = np.array(predictions_all)
        max_probs = np.array(max_probs)
        class_probs_all = np.array(class_probs_all)
        
        # Fit the combiner using the classifier probabilities, human predictions, and truths
        self.combiner.fit(class_probs_all, hum_preds_all, truths_all)
        
        # Fit NearestNeighbors models for each demographic group (with 2 neighbors)
        self.nn_dem0 = NearestNeighbors(n_neighbors=20).fit(features_dem0)
        self.nn_dem1 = NearestNeighbors(n_neighbors=20).fit(features_dem1)
        
        # Store labels and predictions for later retrieval
        self.nn_dem0_labels = labels_dem0
        self.nn_dem0_preds = preds_dem0
        self.nn_dem1_labels = labels_dem1
        self.nn_dem1_preds = preds_dem1

    def compute_statistical_disparity(self, points):
        """
        Compute the accuracy disparity between the 20 closest neighbors from each demographic group
        near the given point(s).

        Parameters:
            points (numpy.ndarray or list of numpy.ndarray): A single feature vector or a list
                of feature vectors. Each feature vector should be 1D.

        Returns:
            np.ndarray: Array of absolute accuracy differences (|accuracy_group0 - accuracy_group1|)
                        for each input point.
        """
        # Ensure points is iterable
        if not isinstance(points, list):
            points = [points]
            
        disparities = []
        for point in points:
            # Reshape the point to 2D (needed for kneighbors query)
            point_reshaped = point.reshape(1, -1)
            
            # Query 20 nearest neighbors for each demographic group
            _, indices_dem0 = self.nn_dem0.kneighbors(point_reshaped, n_neighbors=20)
            _, indices_dem1 = self.nn_dem1.kneighbors(point_reshaped, n_neighbors=20)
            
            # Retrieve predictions and true labels for the neighbors
            # (Assuming self.y_train stores true labels and self.y_pred_train stores model predictions)
            y_pred_dem0 = np.array(self.y_pred_train)[indices_dem0[0]]
            y_true_dem0 = np.array(self.y_train)[indices_dem0[0]]
            y_pred_dem1 = np.array(self.y_pred_train)[indices_dem1[0]]
            y_true_dem1 = np.array(self.y_train)[indices_dem1[0]]
            
            # Compute accuracy for each group
            acc_dem0 = np.mean(y_pred_dem0 == y_true_dem0)
            acc_dem1 = np.mean(y_pred_dem1 == y_true_dem1)
            
            # Compute and store the absolute difference in accuracy
            disparity = abs(acc_dem0 - acc_dem1)
            disparities.append(disparity)
        
        return np.array(disparities)
    def fit(self,dataloader_train,dataloader_val,dataloader_test,epochs,optimizer,lr,verbose=True,test_interval=5,scheduler=None,):
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
        """
        Test the model and compute local statistical parity for each instance based on the 20 nearest
        training neighbors in each demographic group. For each test instance:
        - Query 20 nearest neighbors from training data for demographic groups 0 and 1.
        - Compute the local accuracy (i.e. proportion of correct predictions) for each group using
            the stored training labels (self.nn_demX_labels) and predictions (self.nn_demX_preds).
        - Define the statistical parity as the absolute difference between these accuracies.
        - If parity > self.r, defer the instance (use combiner outputs);
            otherwise, use the classifier’s outputs.
        
        Returns a dictionary with arrays for deferral flags, labels, classifier outputs, and combined outputs.
        """
        defers_all = []
        max_probs = []
        truths_all = []
        hum_preds_all = []
        predictions_all = []   # classifier-only predictions
        rej_score_all = []     # rejector score (unused here; set to 0)
        class_probs_all = []   # classifier probability vectors
        demographics_all = []
        combined_probs_all = []
        combined_preds_all = []
        
        self.model_class.eval()
        with torch.no_grad():
            for batch, (data_x, data_y, hum_preds, demographics) in enumerate(dataloader):
                data_x = data_x.to(self.device)
                data_y = data_y.to(self.device)
                hum_preds = hum_preds.to(self.device)
                demographics = demographics.to(self.device)
                
                # Get classifier outputs
                outputs_class = self.model_class(data_x)
                outputs_class = F.softmax(outputs_class, dim=1)
                max_class_probs, predicted_class = torch.max(outputs_class.data, 1)
                
                # Get combiner outputs (from your combiner module)
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
                
                # Convert classifier outputs to numpy arrays for per-instance processing.
                outputs_np = outputs_class.cpu().numpy()      # probability vectors
                predicted_class_np = predicted_class.cpu().numpy()  # predicted classes
                
                # Process each instance in the current batch.
                batch_defers = []
                batch_combined_probs = []
                batch_combined_preds = []
                for i in range(len(data_y)):
                    # Use the same feature representation as when fitting the NN models (assumed to be data_x)
                    point = data_x[i].cpu().numpy().reshape(1, -1)
                    
                    # Query 20 nearest neighbors for each demographic group from training data.
                    _, indices_dem0 = self.nn_dem0.kneighbors(point, n_neighbors=20)
                    _, indices_dem1 = self.nn_dem1.kneighbors(point, n_neighbors=20)
                    
                    # Retrieve training predictions and true labels for each group.
                    group0_preds = np.array(self.nn_dem0_preds)[indices_dem0[0]]
                    group0_labels = np.array(self.nn_dem0_labels)[indices_dem0[0]]
                    group1_preds = np.array(self.nn_dem1_preds)[indices_dem1[0]]
                    group1_labels = np.array(self.nn_dem1_labels)[indices_dem1[0]]
                    
                    # Compute local accuracy for each group (if no neighbors, default to 0).
                    acc0 = np.mean(group0_preds == group0_labels) if len(group0_preds) > 0 else 0
                    acc1 = np.mean(group1_preds == group1_labels) if len(group1_preds) > 0 else 0
                    
                    # Compute statistical parity as the absolute difference between local accuracies.
                    parity = abs(acc0 - acc1)
                    
                    # If parity exceeds the threshold, defer (i.e. use combiner outputs).
                    if parity > self.r:
                        batch_defers.append(1)
                        # print(f"combined_probs shape: {combined_probs.shape}, index: {i}, y shape: {data_y.shape}")
                        batch_combined_probs.append(combined_probs[i])
                        batch_combined_preds.append(hum_preds[i].cpu().item())
                    else:
                        batch_defers.append(0)
                        batch_combined_probs.append(outputs_np[i])
                        batch_combined_preds.append(predicted_class_np[i])
                
                # Accumulate batch-level results.
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
        
        # Convert lists to numpy arrays.
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
