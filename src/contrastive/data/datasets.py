import numpy as np
np.random.seed(42)
import random
random.seed(42)

import pandas as pd

from pathlib import Path
import glob
import gzip
import pickle
from copy import deepcopy

import torch
import torch.nn.functional as F

from transformers import AutoTokenizer, AutoConfig

from sklearn.preprocessing import LabelEncoder

from pdb import set_trace

class ContrastivePretrainDataset(torch.utils.data.Dataset):
    def __init__(self, path, deduction_set, tokenizer='huawei-noah/TinyBERT_General_4L_312D', max_length=128, intermediate_set=None, clean=False, dataset='lspc', only_interm=False):

        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer, additional_special_tokens=('lenovo','thinkpad','elitebook', 'toshiba', 'asus', 'acer', 'lexar', 'sandisk', 'tesco', 'intenso', 'transcend'))
        self.dataset = dataset

        data = pd.read_pickle(path)
                
        if intermediate_set is not None:
            interm_data = pd.read_pickle(intermediate_set)
            if dataset != 'lspc':
                interm_data['cluster_id'] = interm_data['cluster_id']+10000
            if only_interm:
                data = interm_data
            else:
                data = data.append(interm_data)
        
        data = data.reset_index(drop=True)

        data = data.fillna('')
        data = self._prepare_data(data)

        self.data = data


    def __getitem__(self, idx):
        example = self.data.loc[idx].copy()
        selection = self.data[self.data['labels'] == example['labels']]
        pos = selection.sample(1).iloc[0].copy()

        return (example, pos)

    def __len__(self):
        return len(self.data)
    
    def _prepare_data(self, data):

        label_enc = LabelEncoder()
        data['labels'] = label_enc.fit_transform(data['cluster_id'])

        self.label_encoder = label_enc

        data = data[['features', 'labels']]

        return data