import torch
import os
import random
from sklearn.preprocessing import StandardScaler, LabelEncoder
from torch.utils.data import random_split, TensorDataset, DataLoader

# import loggingc

import pandas as pd
# import numpy as np
# import matplotlib.pyplot as pyplot

from .basedataset import *
import sys

sys.path.append("../")

from .human import *

class Adult(BaseDataset):
    """Adult dataset with Synthetic Humans"""
    def __init__(self, data_dir, device, test_split=0.2, val_split=0.1, batch_size=1000):
        """https://www.kaggle.com/datasets/uciml/adult-census-income"""
        self.data_dir = data_dir
        self.test_split = test_split
        self.val_split = val_split
        self.batch_size = batch_size
        self.n_dataset = 2
        self.train_split = 1 - test_split - val_split
        self.device = device
        self.generate_data()
    def generate_data(self):

        self.random_seed = 42 # change here for ensuring consistency (TODO: add it in __init__ callable during instance creation)

        # df = pd.read_csv(self.data_dir + "/adult.csv")
        df = pd.read_csv(self.data_dir + "/adult_reconstruction.csv")
        df = df.drop(['capital-gain', 'capital-loss'], axis=1)
        # df = df.drop(['educational-num', 'capital-gain', 'capital-loss'], axis=1)

        label_encoder = LabelEncoder()
        standard_scaler = StandardScaler()

        df['gender'] = label_encoder.fit_transform(df['gender'])
        df['workclass'] = label_encoder.fit_transform(df['workclass'])
        df['education'] = label_encoder.fit_transform(df['education'])
        df['marital-status'] = label_encoder.fit_transform(df['marital-status'])
        df['occupation'] = label_encoder.fit_transform(df['occupation'])
        df['relationship'] = label_encoder.fit_transform(df['relationship'])
        df['race'] = label_encoder.fit_transform(df['race'])
        df['native-country'] = label_encoder.fit_transform(df['native-country'])
        df['income'] = label_encoder.fit_transform(df['income'])

        # y = df['income']
        # X = df.drop(['income', 'gender'], axis=1) # can remove gender if I wanna include it as a feature in dataset
        # demographics = df['gender']

        # X_train, X_test, y_train, y_test, d_train, d_test = train_test_split(X, y, gender, test_size=0.3, random_state = self.random_seed)

        # X_train = st.fit_transform(X_train)
        # X_test = st.transform(X_test)

        t1, t2 = 25_000, 60_000 #28428.48963703, 64663.35819187 are cluster thresholds according to K-means
        y = df['income'].to_numpy()
        y = np.where(y < t1, 0, np.where(y < t2, 1, 2))
        X = df.drop(['income', 'gender'], axis=1).to_numpy()  # can remove gender if I wanna include it as a feature in dataset
        demographics = df['gender'].to_numpy()

        # Normalize data
        X = standard_scaler.fit_transform(X)

        # Convert to torch tensors
        X = torch.tensor(X, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.int64)
        demographics = torch.tensor(demographics, dtype=torch.int64)
        # expert = torch.tensor(make_biased_humans(y, demographics, accuracy=0.8, odds_diff=0)) # change it to consider label set later
        expert = torch.tensor(exact_fair_hatespeech(y))
        self.d = X.shape[1]

        # Split data
        total_samples = len(X)
        train_size = int(self.train_split * total_samples)
        val_size = int(self.val_split * total_samples)
        test_size = total_samples - train_size - val_size

        random_seed = random.randrange(10000)
        train_data, val_data, test_data = random_split(
            TensorDataset(X, y, expert, demographics),
            [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(random_seed)
        )

        self.data_train_loader = DataLoader(train_data, batch_size=self.batch_size, shuffle=True)
        self.data_val_loader = DataLoader(val_data, batch_size=self.batch_size, shuffle=True)
        self.data_test_loader = DataLoader(test_data, batch_size=self.batch_size, shuffle=True)

        

























