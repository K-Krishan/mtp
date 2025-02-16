import torch
import os
import random
from sklearn.preprocessing import StandardScaler, LabelEncoder

import logging

import pandas as pd
import numpy as np
import matplotlib.pyplot as pyplot

sys.path.append("../")

from .human import *

class AdultDataset(BaseDataset):
    """Adult dataset with Synthetic Humans"""
    def __init__(self, data_dir, test_split=0.2, val_split=0.1, batch_size=1000):
        """https://www.kaggle.com/datasets/uciml/adult-census-income"""
        self.data_dir = data_dir
        self.test_split = test_split
        self.val_split = val_split
        self.batch_size = batch_size
        self.n_dataset = 2
        self.train_split = 1 - test_split - val_split
        self.generate_data()
    def generate_data(self):

        self.random_seed = 42 # change here for ensuring consistency (TODO: add it in __init__ callable during instance creation)

        adult_data = pd.read_csv(self.data_dir + "/adult.csv")
        df = df.drop(['educational-num', 'capital-gain', 'capital-loss'], axis=1)

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
        
        y = df['income'].to_numpy()
        X = df.drop(['income', 'gender'], axis=1).to_numpy()  # can remove gender if I wanna include it as a feature in dataset
        demographics = df['gender'].to_numpy()

        # Normalize data
        X = standard_scaler.fit_transform(X)

        # Convert to torch tensors
        X = torch.tensor(X, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.int64)
        demographics = torch.tensor(demographics, dtype=torch.int64)
        expert = make_biased_humans(y, demographics, accuracy=0.8, odds_diff=0) # change it to consider label set later

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

        

























