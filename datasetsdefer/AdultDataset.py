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

import torch
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.utils.data import DataLoader, TensorDataset, random_split
from .basedataset import BaseDataset
import torch
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.utils.data import DataLoader, TensorDataset, random_split
from .basedataset import BaseDataset

class Adult(BaseDataset):
    """Adult Census Income Dataset"""
    def __init__(self, data_dir, device, test_split=0.2, val_split=0.1, batch_size=1000):
        super().__init__()
        self.data_dir = data_dir
        self.device = device
        self.test_split = test_split
        self.val_split = val_split
        self.batch_size = batch_size
        self.n_dataset = 3  # Number of income classes
        self.generate_data()

    def generate_data(self):
        # Load and preprocess data
        df = pd.read_csv(self.data_dir + "/adult_reconstruction.csv")
        
        # Process income labels
        t1, t2 = 25000, 60000
        df['income'] = pd.cut(df['income'], 
                            bins=[0, t1, t2, float('inf')], 
                            labels=[0, 1, 2])
        
        # Encode categorical features including gender
        categorical_cols = ['workclass', 'education', 'marital-status',
                           'occupation', 'relationship', 'race', 'native-country', 'gender']
        self.encoders = {col: LabelEncoder() for col in categorical_cols}
        
        for col in categorical_cols:
            df[col] = self.encoders[col].fit_transform(df[col])

        # Extract demographics (encoded gender)
        demographics = df['gender'].values
        y = df['income'].values
        X = df.drop(['income', 'gender'], axis=1).values

        # Normalize features
        self.scaler = StandardScaler()
        X = self.scaler.fit_transform(X)

        # Create synthetic human labels
        human_labels = self._create_human_labels(y, accuracy=0.65)

        # Convert to tensors - ensure all arrays are numeric
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.LongTensor(y.astype('int64'))
        human_tensor = torch.LongTensor(human_labels.astype('int64'))
        demo_tensor = torch.LongTensor(demographics.astype('int64'))

        # Create dataset and split
        full_dataset = TensorDataset(X_tensor, y_tensor, human_tensor, demo_tensor)
        
        total_size = len(full_dataset)
        test_size = int(total_size * self.test_split)
        val_size = int(total_size * self.val_split)
        train_size = total_size - test_size - val_size
        
        self.train_dataset, self.val_dataset, self.test_dataset = random_split(
            full_dataset, [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(42)
        )
        
        # Set required BaseDataset attributes
        self.d = X.shape[1]  # Feature dimension
        self.n_dataset = 3   # Number of classes
        
        # Create DataLoaders
        pin_memory = (self.device.type == 'cuda')
        self.data_train_loader = DataLoader(
            self.train_dataset, 
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=pin_memory
        )
        
        self.data_val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=pin_memory
        )
        
        self.data_test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=pin_memory
        )

    def _create_human_labels(self, y, accuracy=0.8):
        """Create synthetic human labels with specified accuracy"""
        human_labels = np.copy(y)
        mask = np.random.rand(len(y)) > accuracy
        
        # For incorrect labels, randomly choose from other classes
        for i in np.where(mask)[0]:
            possible_labels = [l for l in np.unique(y) if l != y[i]]
            human_labels[i] = np.random.choice(possible_labels)
            
        return human_labels
