from tqdm.auto import tqdm
import torch
from sklearn.model_selection import train_test_split
import numpy as np
import warnings
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV, StratifiedKFold

import numpy as np

from torch import nn, optim
from torch.nn.functional import softmax
from sklearn.cluster import KMeans

class TSCalibrator():
    """ Maximum likelihood temperature scaling (Guo et al., 2017)
    """

    def __init__(self, temperature=1., n_clusters=14):
        super().__init__()
        self.temperature = []
        self.n_clusters = n_clusters
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=0)

        self.loss_trace = None

    def fitHelper(self, logits, y, cluster):
        """ Fits temperature scaling using hard labels.
        """
        # Pre-processing
        self.n_classes = logits.shape[1]
        _model_logits = torch.from_numpy(logits)
        _y = torch.from_numpy(y)
        _temperature = torch.tensor(1.0, requires_grad=True)

        # Optimization parameters
        nll = nn.CrossEntropyLoss()  # Supervised hard-label loss
        num_steps = 7500
        learning_rate = 0.05
        grad_tol = 1e-3  # Gradient tolerance for early stopping
        min_temp, max_temp = 1e-2, 1e4  # Upper / lower bounds on temperature

        optimizer = optim.Adam([_temperature], lr=learning_rate)

        loss_trace = []  # Track loss over iterations
        step = 0
        converged = False
        while not converged:

            optimizer.zero_grad()
            loss = nll(_model_logits / _temperature, _y)
            loss.backward()
            optimizer.step()
            loss_trace.append(loss.item())

            with torch.no_grad():
                _temperature.clamp_(min=min_temp, max=max_temp)

            step += 1
            if step > num_steps:
                warnings.warn('Maximum number of steps reached -- may not have converged (TS)')
            converged = (step > num_steps) or (np.abs(_temperature.grad) < grad_tol)

        self.loss_trace = loss_trace
        # self.temperature = _temperature.item()
        return _temperature.item()

    def fit(self, model_logits, y_true):
        self.kmeans.fit(model_logits)

        _model_logits = []
        _y_true = []
        for _ in range(self.n_clusters):
            _model_logits.append([])
            _y_true.append([])
            self.temperature.append(1.0)

        for i in range(self.kmeans.labels_.shape[0]):
            cluster_center = self.kmeans.labels_[i]
            _model_logits[cluster_center].append(model_logits[i])
            _y_true[cluster_center].append(y_true[i])

        for cluster in range(0, self.n_clusters):
            self.temperature[cluster] = self.fitHelper(np.array(_model_logits[cluster]), np.array(_y_true[cluster]), cluster)

    def calibrate(self, probs):
        clipped_model_probs = np.clip(probs, 1e-50, 1)
        model_logits = np.log(clipped_model_probs)
        labels = self.kmeans.predict(model_logits)

        for i in range(model_logits.shape[0]):
            probs[i] = probs[i] ** (1. / self.temperature[labels[i]])

        # calibrated_probs = probs ** (1. / self.temperature)  # Temper
        probs /= np.sum(probs, axis=1, keepdims=True)  # Normalize
        return probs

class AllCombiner:
    """ Implements the P+L combination method, fit using maximum likelihood
    """
    def __init__(self, calibration_method='temperature scaling', **kwargs):
        self.calibrator = None
        self.confusion_matrix = None  # conf[i, j] is assumed to be P(h = i | Y = j)

        self.n_train_u = None  # Amount of unlabeled training data
        self.n_train_l = None  # Amount of labeled training data
        self.n_cls = None  # Number of classes

        self.eps = 1e-50

        self.use_cv = False
        self.calibration_method = calibration_method
        self.calibrator = TSCalibrator()

    def calibrate(self, model_probs):
        return self.calibrator.calibrate(model_probs)

    def fit(self, model_probs, y_h, y_true):
        self.n_cls = model_probs.shape[1]

        # Estimate human confusion matrix
        # Entry [i, j]  is #(Y = i and h = j)
        conf_h = 1. * confusion_matrix(y_true, y_h, labels=np.arange(self.n_cls))
        # Swap so entry [i, j] is #(h = i and Y = j)
        conf_h = conf_h.T
        conf_h = np.clip(conf_h, self.eps, None)
        normalizer = np.sum(conf_h, axis=0, keepdims=True)
        # Normalize columns so entry [i, j] is P(h = i | Y = j)
        conf_h /= normalizer
        self.confusion_matrix = conf_h

        # Calibrate model probabilities
        if self.use_cv:
            self.fit_calibrator_cv(model_probs, y_true)
        else:
            self.fit_calibrator(model_probs, y_true)

    def fit_bayesian(self, model_probs, y_h, y_true, alpha=0.1, beta=0.1):
        """ This is the "plus one" parameterization, i.e. alpha,beta just need to be > 0
        Really corresponds to a Dirichlet(alpha+1, beta+1, beta+1, . . . ,beta+1) distribution
        """
        self.n_cls = model_probs.shape[1]

        prior_matr = np.eye(self.n_cls) * alpha + (np.ones(self.n_cls) - np.eye(self.n_cls)) * beta

        conf_h = 1. * confusion_matrix(y_true, y_h, labels=np.arange(self.n_cls))
        conf_h += prior_matr
        # Swap so entry [i, j] is #(h = i and Y = j)
        conf_h = conf_h.T
        #conf_h = np.clip(conf_h, self.eps, None)
        normalizer = np.sum(conf_h, axis=0, keepdims=True)
        # Normalize columns so entry [i, j] is P(h = i | Y = j)
        conf_h = conf_h / normalizer
        self.confusion_matrix = conf_h

        # Calibrate model probabilities
        if self.use_cv:
            self.fit_calibrator_cv(model_probs, y_true)
        else:
            self.fit_calibrator(model_probs, y_true)

    def fit_calibrator(self, model_probs, y_true):
        clipped_model_probs = np.clip(model_probs, self.eps, 1)
        model_logits = np.log(clipped_model_probs)
        self.calibrator.fit(model_logits, y_true)

    def fit_calibrator_cv(self, model_probs, y_true):
        # Fits calibration maps that require hyperparameters, using cross-validation
        if self.calibration_method == 'dirichlet':
            reg_lambda_vals = [10., 1., 0., 5e-1, 1e-1, 1e-2, 1e-3]
            skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=0)
            gscv = GridSearchCV(self.calibrator, param_grid={'reg_lambda': reg_lambda_vals,
                                                             'reg_mu': [None]},
                                cv=skf, scoring='neg_log_loss', refit=True)
            gscv.fit(model_probs, y_true)
            self.calibrator = gscv.best_estimator_
        else:
            raise NotImplementedError

    def combine_proba(self, model_probs, y_h, y_true_te):
        """ Combines model probabilities with hard labels via the calibrate-confuse equation given the confusion matrix.

        Args:
            p_m: Array of model probabilities ; shape (n_samples, n_classes)
            y_h: List of hard labels ; shape (n_samples,)

        Returns:
            Normalized posterior probabilities P(Y | m, h). Entry [i, j] is P(Y = j | h_i, m_i)
        """
        assert model_probs.shape[0] == y_h.size, 'Size mismatch between model probs and human labels'
        assert model_probs.shape[1] == self.n_cls, 'Size mismatch between model probs and number of classes'

        n_samples = model_probs.shape[0]
        calibrated_model_probs = self.calibrate(model_probs)

        model_output = np.argmax(calibrated_model_probs, axis=1)
        
        # Accuracy Parameters
        human_correct = 0
        human_refered = 0
        model_correct = 0
        model_refered = 0
        
        y_comb = np.empty((n_samples, self.n_cls))
        for i in range(n_samples):

            # P_X = miss_cost * (1 - calibrated_model_probs[i][model_output[i]])
            # if(True):
                # y_comb[i] = calibrated_model_probs[i] * self.confusion_matrix[y_h[i]]
            y_comb[i] = calibrated_model_probs[i] * self.confusion_matrix[y_h[i]]
            human_refered += 1
            if(np.argmax(y_comb[i]) == y_true_te[i]):
                human_correct += 1

            if np.allclose(y_comb[i], 0):  # Handle zero rows
                y_comb[i] = np.ones(self.n_cls) * (1./self.n_cls)

        result = {
            # 'Missclassification Cost' : miss_cost,
            # 'Human cost': human_cost,
            # 'Combined accuracy': (human_correct + model_correct) / n_samples,
            # 'Human correct': human_correct,
            # 'Human refered': human_refered,
            # 'Human accuracy': human_correct / human_refered,
            # 'Model correct': model_correct,
            # 'Model refered': model_refered,
            # 'Model accuracy': model_correct / model_refered,
            # 'Model cost': miss_cost * (model_refered - model_correct) + human_cost * human_refered
        }

        # Don't forget to normalize :)
        assert np.all(np.isfinite(np.sum(y_comb, axis=1)))
        assert np.all(np.sum(y_comb, axis=1) > 0)
        y_comb /= np.sum(y_comb, axis=1, keepdims=True)
        return y_comb, result
        # return calibrated_model_probs, self.confusion_matrix, y_comb

    def combine(self, model_probs, y_h, y_true_te):
        """ Combines model probs and y_h to return hard labels
        """
        y_comb_soft, result = self.combine_proba(model_probs, y_h, y_true_te)
        return np.argmax(y_comb_soft, axis=1), result

        # calibrated_model_probs, confusion_matrix, y_comb_soft = self.combine_proba(model_probs, y_h)
        # return calibrated_model_probs, confusion_matrix, y_comb_soft
