import torch
import os
import random
import sys
from sklearn.preprocessing import StandardScaler
import torch
import pandas as pd
import logging
import sys

sys.path.append("../")
import zipfile
import requests
from .basedataset import BaseDataset
from .human import *

def synth(y, d, synth=[0.8,0.6]):
    h = []
    for i in range(len(y)):
        if np.random.rand() < synth[d[i]]:
            h.append(y[i])
        else:
            h.append(1-y[i]) # change it to consider label set later
    return np.array(h)

def biased_synth_multiple_demographics(y, d, sensitive_label, accuracy=0.8, odds_diff=0.2):
    demographic_count = np.max(d)+1 # assuming 0 to d
    probs = []
    if odds_diff == 0:
        for i in range(demographic_count):
            probs.append(accuracy)
    else:
        for i in range(demographic_count):
            if i == sensitive_label:
                probs.append(accuracy - odds_diff*(demographic_count-1)/demographic_count)
            else:
                probs.append(accuracy + odds_diff/demographic_count)
    return synth(y, d, synth=probs)

class BrowardDataset(BaseDataset):
    """Compas dataset with human judgements for 1000 points"""

    def __init__(
        self, data_dir, test_split=0.2, val_split=0.1, batch_size=1000, transforms=None
    ):
        """
        https://farid.berkeley.edu/downloads/publications/scienceadvances17/allData.zip
        https://www.science.org/doi/10.1126/sciadv.aao5580

        data_dir: where to save files for model
        test_split: percentage of test data
        val_split: percentage of data to be used for validation (from training set)
        batch_size: batch size for training
        transforms: data transforms
        """
        self.data_dir = data_dir
        self.test_split = test_split
        self.val_split = val_split
        self.batch_size = batch_size
        self.n_dataset = 2
        self.train_split = 1 - test_split - val_split
        self.transforms = transforms
        self.generate_data()

    def generate_data(self):
        """
        generate data for training, validation and test sets
        """
        logging.info("Loading Broward data")
        try:
            broward_data = pd.read_csv(
                self.data_dir + "/allDataBroward/BROWARD_CLEAN_SUBSET.csv"
            )
            mturk_data = pd.read_csv(
                self.data_dir + "/allDataBroward/MTURK_RACE.csv"
            )
        except:
            logging.error("Failed to load Broward data")
            raise

        broward_data = broward_data.drop(["block_num", "id"], axis=1)
        train_y = broward_data.two_year_recid.to_numpy()
        broward_data = broward_data.drop(["two_year_recid"], axis=1)
        train_x = broward_data.to_numpy()
        demographics = broward_data["race"].to_numpy()

        # normalize data
        scaler = StandardScaler()
        scaler.fit(train_x)
        train_x = scaler.transform(train_x)

        # human_predictions = []
        # mturk_data = mturk_data.drop(["mTurk_code"], axis=1)
        # for i in range(1, len(mturk_data)):
        #     # get all columns
        #     row = mturk_data.iloc[i]
        #     # only keep the columns that are not nan
        #     row = row[row.notna()]
        #     # # way1: choose a random human
        #     # random_sample = row.sample(n=1).values[0]
        #     # most_common = row.value_counts().idxmax()

        #     # # way2: choosing correct label based on number of correct humans
        #     # if most_common == 1:
        #     #     human_predictions.append(train_y[i - 1])
        #     # else:
        #     #     human_predictions.append(1 - train_y[i - 1])

        #     # way3: choose label with probabilities
        #     label_counts = row.value_counts()
        #     total_votes = sum(label_counts)
        #     prob_correct = label_counts.get(1, 0) / total_votes  # P(correct) = ratio of '1' votes
        #     prob_wrong = 1 - prob_correct  # P(wrong) = remaining probability
        #     # Sample the label based on probability
        #     sampled_label = np.random.choice(
        #         [train_y[i-1], 1 - train_y[i-1]],  # Options: correct or flipped
        #         p=[prob_correct, prob_wrong]   # Probabilities
        #     )
        #     human_predictions.append(sampled_label)

        # human_predictions = torch.tensor(human_predictions)
        human_predictions = torch.tensor(biased_synth_multiple_demographics(train_y, demographics, 2, 0.7, 0.2))

        train_y = torch.tensor(train_y)
        demographics = torch.from_numpy(demographics).int()
        train_x = torch.from_numpy(train_x).float()
        self.total_samples = len(train_x)
        self.d = len(train_x[0])

        # print(human_predictions.shape, train_y.shape, demographics.shape, train_x.shape)

        # change seed here for experimentation
        random_seed = random.randrange(10000)

        train_size = int(self.train_split * self.total_samples)
        val_size = int(self.val_split * self.total_samples)
        test_size = self.total_samples - train_size - val_size
        self.train_x, self.val_x, self.test_x = torch.utils.data.random_split(
            train_x,
            [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(random_seed),
        )
        self.train_y, self.val_y, self.test_y = torch.utils.data.random_split(
            train_y,
            [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(random_seed),
        )
        self.train_h, self.val_h, self.test_h = torch.utils.data.random_split(
            human_predictions,
            [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(random_seed),
        )
        self.train_d, self.val_d, self.test_d = torch.utils.data.random_split(
            demographics,
            [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(random_seed),
        )

        logging.info(f"train size:  {len(self.train_x)}")
        logging.info(f"val size: {len(self.val_x)}")
        logging.info(f"test size:  {len(self.test_x)}")


        self.data_train = torch.utils.data.TensorDataset(
            self.train_x.dataset.data[self.train_x.indices],
            self.train_y.dataset.data[self.train_y.indices],
            self.train_h.dataset.data[self.train_h.indices],
            self.train_d.dataset.data[self.train_d.indices],
        )
        self.data_val = torch.utils.data.TensorDataset(
            self.val_x.dataset.data[self.val_x.indices],
            self.val_y.dataset.data[self.val_y.indices],
            self.val_h.dataset.data[self.val_h.indices],
            self.val_d.dataset.data[self.val_d.indices],
        )
        self.data_test = torch.utils.data.TensorDataset(
            self.test_x.dataset.data[self.test_x.indices],
            self.test_y.dataset.data[self.test_y.indices],
            self.test_h.dataset.data[self.test_h.indices],
            self.test_d.dataset.data[self.test_d.indices],
        )

        self.data_train_loader = torch.utils.data.DataLoader(
            self.data_train, batch_size=self.batch_size, shuffle=True
        )
        self.data_val_loader = torch.utils.data.DataLoader(
            self.data_val, batch_size=self.batch_size, shuffle=True
        )
        self.data_test_loader = torch.utils.data.DataLoader(
            self.data_test, batch_size=self.batch_size, shuffle=True
        )
