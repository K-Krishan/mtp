import torch
import numpy as np
import os
import random
import sys
import torch
import logging
import sys

import torch.utils

sys.path.append("../")
import requests
# from torchtext import data
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from .basedataset import BaseDataset
from .human import *

import re

def check_embeddings(file_path, train_x, model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"):
    """
    Load embeddings from file if they exist, otherwise generate and save them.

    Args:
        file_path (str): Path to save/load the embeddings.
        train_x (list of str): Text data to embed.
        model_name (str): SentenceTransformer model name.

    Returns:
        np.ndarray: Embeddings as a NumPy array.
    """
    if os.path.exists(file_path):
        logging.info(f"Loading cached embeddings from {file_path}")
        embeddings = np.load(file_path)
    else:
        logging.info("Generating embeddings")
        model = SentenceTransformer(model_name)
        embeddings = model.encode(train_x)
        logging.info(f"Saving embeddings to {file_path}")
        np.save(file_path, embeddings)
    return embeddings

class ModelPredictAAE:
    def __init__(self, modelfile, vocabfile):
        """
        Edited from https://github.com/slanglab/twitteraae
        """
        self.vocabfile = vocabfile
        self.modelfile = modelfile
        self.load_model()

    def load_model(self):

        self.N_wk = np.loadtxt(self.modelfile)
        self.N_w = self.N_wk.sum(1)
        self.N_k = self.N_wk.sum(0)
        self.K = len(self.N_k)
        self.wordprobs = (self.N_wk + 1) / self.N_k
        self.vocab = [
            L.split("\t")[-1].strip() for L in open(self.vocabfile, encoding="utf8")
        ]
        self.w2num = {w: i for i, w in enumerate(self.vocab)}
        assert len(self.vocab) == self.N_wk.shape[0]

    def infer_cvb0(self, invocab_tokens, alpha, numpasses):
        doclen = len(invocab_tokens)

        # initialize with likelihoods
        Qs = np.zeros((doclen, self.K))
        for i in range(0, doclen):
            w = invocab_tokens[i]
            Qs[i, :] = self.wordprobs[self.w2num[w], :]
            Qs[i, :] /= Qs[i, :].sum()
        lik = Qs.copy()  # pertoken normalized but proportionally the same for inference

        Q_k = Qs.sum(0)
        for itr in range(1, numpasses):
            # print "cvb0 iter", itr
            for i in range(0, doclen):
                Q_k -= Qs[i, :]
                Qs[i, :] = lik[i, :] * (Q_k + alpha)
                Qs[i, :] /= Qs[i, :].sum()
                Q_k += Qs[i, :]

        Q_k /= Q_k.sum()
        return Q_k

    def predict_lang(self, tokens, alpha=1, numpasses=5, thresh1=1, thresh2=0.2):
        invocab_tokens = [w.lower() for w in tokens if w.lower() in self.w2num]
        # check that at least xx tokens are in vocabulary
        if len(invocab_tokens) < thresh1:
            return 0
        # check that at least yy% of tokens are in vocabulary
        elif len(invocab_tokens) / len(tokens) < thresh2:
            return 0
        else:
            posterior = self.infer_cvb0(
                invocab_tokens, alpha=alpha, numpasses=numpasses
            )
            # posterior is probability for African-American, Hispanic, Asian, and White (in that order)
            aae = (np.argmax(posterior) == 0) * 1
            return aae

def custom_split(tweet):
    # Replace double quotes with a specific delimiter or spaces
    tweet = tweet.replace('"', ' " ')
    
    # Optionally, replace all non-alphanumeric characters (except spaces) with spaces
    tweet = re.sub(r'[^A-Za-z0-9\s]', ' ', tweet)
    
    # Now split the tweet
    return tweet.split()

# https://github.com/jcpeterson/cifar-10h
class HateSpeech(BaseDataset):
    """ Hatespeech dataset from Davidson et al. 2017 """
    def __init__(
        self,
        data_dir,
        embed_texts,
        include_demographics,
        expert_type,
        device,
        synth_exp_param=[0.7, 0.7],
        test_split=0.2,
        val_split=0.1,
        batch_size=1000,
        transforms=None,
    ):
        """
        data_dir: where to save files for dataset (folder path)
        embed_texts (bool): whether to embedd the texts or raw text return
        include_demographics (bool): whether to include the demographics for each example, defined as either AA or not.
        if True, then the data loader will return a tuple of (data, label, expert_prediction, demographics)
        expert_type (str): either 'synthetic' which makes error depending on synth_exp_param for not AA or AA, or 'random_annotator' which defines human as random annotator
        synth_exp_param (list): list of length 2, first element is the probability of error for AA, second is for not AA
        test_split: percentage of test data
        val_split: percentage of data to be used for validation (from training set)
        batch_size: batch size for training
        transforms: data transforms
        """
        self.embed_texts = embed_texts
        self.include_demographics = include_demographics
        self.expert_type = expert_type
        self.synth_exp_param = synth_exp_param
        self.data_dir = data_dir
        self.device = device
        self.test_split = test_split
        self.val_split = val_split
        self.batch_size = batch_size
        self.n_dataset = 3  # number of classes in dataset
        self.train_split = 1 - test_split - val_split
        self.transforms = transforms
        self.generate_data()

    def generate_data(self):
        """
        generate data for training, validation and test sets

        """
        # download dataset if it doesn't exist
        if not os.path.exists(self.data_dir + "/hatespeech_labeled_data.csv"):
            logging.info("Downloading HateSpeech dataset")
            r = requests.get(
                "https://github.com/t-davidson/hate-speech-and-offensive-language/raw/master/data/labeled_data.csv",
                allow_redirects=True,
            )
            with open(self.data_dir + "/hatespeech_labeled_data.csv", "wb") as f:
                f.write(r.content)
            logging.info("Finished Downloading HateSpeech Data data")
            try:
                hatespeech_data = pd.read_csv(
                    self.data_dir + "/hatespeech_labeled_data.csv"
                )
            except:
                logging.error("Failed to load HateSpeech data")
                raise
        else:
            logging.info("Loading HateSpeech data")
            try:
                hatespeech_data = pd.read_csv(
                    self.data_dir + "/hatespeech_labeled_data.csv"
                )
            except:
                logging.error("Failed to load HateSpeech data")
                raise
            # download aae file
        if not os.path.exists(self.data_dir + "/model_count_table.txt"):
            logging.info("Downloading AAE detection")
            r = requests.get(
                "https://github.com/slanglab/twitteraae/raw/master/model/model_count_table.txt",
                allow_redirects=True,
            )
            with open(self.data_dir + "/model_count_table.txt", "wb") as f:
                f.write(r.content)
        if not os.path.exists(self.data_dir + "/model_vocab.txt"):
            logging.info("Downloading AAE detection")
            r = requests.get(
                "https://github.com/slanglab/twitteraae/raw/master/model/model_vocab.txt",
                allow_redirects=True,
            )
            with open(self.data_dir + "/model_vocab.txt", "wb") as f:
                f.write(r.content)
        self.model_file_path = self.data_dir + "/model_count_table.txt"
        self.vocab_file_path = self.data_dir + "/model_vocab.txt"
        self.model_aae = ModelPredictAAE(self.model_file_path, self.vocab_file_path)
        # predict demographics for the deata

        hatespeech_data["demographics"] = hatespeech_data["tweet"].apply(custom_split).apply(
            lambda x: self.model_aae.predict_lang(x)
        )

        self.label_to_category = {
            0: "hate_speech",
            1: "offensive_language",
            2: "neither",
        }
        # create a new column that creates a distribution over the labels
        distribution_over_labels = []
        for i in range(len(hatespeech_data)):
            label_counts = [
                hatespeech_data.iloc[i]["hate_speech"],
                hatespeech_data.iloc[i]["offensive_language"],
                hatespeech_data.iloc[i]["neither"],
            ]
            label_distribution = np.array(label_counts) / sum(label_counts)
            distribution_over_labels.append(label_distribution)
        hatespeech_data["label_distribution"] = distribution_over_labels

        # change the distribution here if i need
        human_prediction = exact_fair_hatespeech(hatespeech_data["class"])
        # human_prediction = []
        # if self.expert_type == "synthetic":
        #     for i in range(len(hatespeech_data)):
        #         if hatespeech_data.iloc[i]["demographics"] == 0:
        #             correct_human = np.random.choice(
        #                 [0, 1], p=[1 - self.synth_exp_param[0], self.synth_exp_param[0]]
        #             )

        #         else:
        #             correct_human = np.random.choice(
        #                 [0, 1], p=[1 - self.synth_exp_param[1], self.synth_exp_param[1]]
        #             )
        #         if correct_human:
        #             human_prediction.append(hatespeech_data.iloc[i]["class"])
        #         else:
        #             human_prediction.append(np.random.choice([0, 1, 2]))
        # else:
        #     for i in range(len(hatespeech_data)):
        #         # sample from label distribution
        #         label_distribution = hatespeech_data.iloc[i]["label_distribution"]
        #         label = np.random.choice([0, 1, 2], p=label_distribution)
        #         human_prediction.append(label)

        hatespeech_data["human_prediction"] = human_prediction

        train_x = hatespeech_data["tweet"].to_numpy()
        train_y = hatespeech_data["class"].to_numpy()
        train_h = hatespeech_data["human_prediction"].to_numpy()
        train_d = hatespeech_data["demographics"].to_numpy()
        random_seed = random.randrange(10000)
        # random_seed = 42 # for consistency during my testing

        logging.info("Embedding texts")
        # TODO: cache the embeddings, so no need to regenerate them
        minilml12v2_embed_path = "hatespeech_embeddings.npy"
        embeddings = check_embeddings(minilml12v2_embed_path,train_x)
        
        train_x = np.array(embeddings)
        test_size = int(self.test_split * len(hatespeech_data))
        val_size = int(self.val_split * len(hatespeech_data))
        train_size = len(hatespeech_data) - test_size - val_size
        train_y = torch.tensor(train_y)
        train_h = torch.tensor(train_h)
        train_d = torch.tensor(train_d)
        train_x = torch.from_numpy(train_x).float()

        self.d = train_x.shape[1]
        train_x, val_x, test_x = torch.utils.data.random_split(
            train_x,
            [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(random_seed),
        )
        train_y, val_y, test_y = torch.utils.data.random_split(
            train_y,
            [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(random_seed),
        )
        train_h, val_h, test_h = torch.utils.data.random_split(
            train_h,
            [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(random_seed),
        )
        train_d, val_d, test_d = torch.utils.data.random_split(
            train_d,
            [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(random_seed),
        )

        data_train = torch.utils.data.TensorDataset(
            train_x.dataset[train_x.indices],
            train_y.dataset[train_y.indices],
            train_h.dataset[train_h.indices],
            train_d.dataset[train_d.indices],
        )
        data_val = torch.utils.data.TensorDataset(
            val_x.dataset[val_x.indices],
            val_y.dataset[val_y.indices],
            val_h.dataset[val_h.indices],
            val_d.dataset[val_d.indices],
        )
        data_test = torch.utils.data.TensorDataset(
            test_x.dataset[test_x.indices],
            test_y.dataset[test_y.indices],
            test_h.dataset[test_h.indices],
            test_d.dataset[test_d.indices],
        )

        self.data_train_loader = torch.utils.data.DataLoader(
            data_train, batch_size=self.batch_size, shuffle=False
        )
        self.data_val_loader = torch.utils.data.DataLoader(
            data_val, batch_size=self.batch_size, shuffle=False
        )
        self.data_test_loader = torch.utils.data.DataLoader(
            data_test, batch_size=self.batch_size, shuffle=False
        )

    def model_setting(self, model_nn):
        # build model
        INPUT_DIM = len(self.text_field.vocab)
        EMBEDDING_DIM = 100  # fixed
        PAD_IDX = self.text_field.vocab.stoi[self.text_field.pad_token]

        # model = CNN(INPUT_DIM, EMBEDDING_DIM, N_FILTERS, FILTER_SIZES, OUTPUT_DIM, DROPOUT, PAD_IDX)
        # model = CNN_rej(INPUT_DIM, EMBEDDING_DIM, N_FILTERS, FILTER_SIZES, 3, DROPOUT, PAD_IDX)

        pretrained_embeddings = self.text_field.vocab.vectors

        model_nn.embedding.weight.data.copy_(pretrained_embeddings)
        UNK_IDX = self.text_field.vocab.stoi[self.text_field.unk_token]

        model_nn.embedding.weight.data[UNK_IDX] = torch.zeros(EMBEDDING_DIM)
        model_nn.embedding.weight.data[PAD_IDX] = torch.zeros(EMBEDDING_DIM)
        return INPUT_DIM, EMBEDDING_DIM, PAD_IDX
