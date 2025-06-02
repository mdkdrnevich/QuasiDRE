import uproot
import numpy as np
import torch
from torch.utils.data import TensorDataset, ConcatDataset, DataLoader
from sklearn import preprocessing
import collections
import zipfile
import yaml
from tqdm import tqdm
import os.path as osp

from .tools import os_splitroot

class Dataset:
    def __init__(
        self,
        file_name,
        label,
        start_event=0,
        stop_event=None,
    ):
        """
        Initialize the dataset
        Args:
            file_name (str): path
            label (int): 0/1 for binary classification
            start_event (int): index of the first data point to process
            stop_event (int): index (exclusive) of the last data point to process
        """
        self.file_name = file_name
        self.label = label
        self.start_event = start_event
        self.stop_event = stop_event
        self.datas = []


    def process(self, normalize_weights=False, weight_norm=None):
        """
        Handles conversion of dataset file at raw_path into PyTorch dataset.
        """
        self.datas = []
        with open(self.file_name, 'rb') as f:
            data = np.load(f)


        if self.stop_event is None:
            self.stop_event = int(data.shape[0])
        self.n_events = self.stop_event - self.start_event
        
        x = torch.tensor(data[self.start_event:self.stop_event, :-1].reshape(self.n_events, -1), dtype=torch.float)
        w = torch.tensor(data[self.start_event:self.stop_event, -1].reshape(self.n_events, 1), dtype=torch.float)
        if normalize_weights is True:
            if weight_norm is None:
                weight_norm = w.mean()
            w = w / weight_norm
            
        if self.label == 0:
            y = torch.zeros((self.n_events, 1), dtype=torch.float)
        elif self.label == 1:
            y = torch.ones((self.n_events, 1), dtype=torch.float)
        else:
            y = torch.ones((self.n_events, 1), dtype=torch.float)*self.label

        self.datas.append(TensorDataset(x, y, w))
        return weight_norm


def CombinedDataset(*args):
    total = []
    for arg in args:
        total.extend(arg.datas)
    return ConcatDataset(total)


def get_scaling(dataset, batch_size=1024, shuffle=False):
    scaling_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    scaling_loader.collate_fn = prep_inputs_for_scaling

    X_scaler = preprocessing.StandardScaler()
    weight_total = 0
    num_total_weights = 0
    
    for batch in tqdm(scaling_loader):
        X_scaler.partial_fit(batch[0])
        weight_total += batch[1].sum()
        num_total_weights += len(batch[1])
    weight_norm = weight_total / num_total_weights
    return X_scaler, weight_norm


def prep_inputs_for_scaling(batch_list):
    x_batch_list = []
    w_batch_list = []
    for sample in batch_list:
        x_batch_list.append(sample[0])
        w_batch_list.append(sample[2])
    x_batch = torch.stack(x_batch_list)
    w_batch = torch.stack(w_batch_list)
    return x_batch, w_batch


def prep_inputs_for_training(batch_list, X_scaler, weight_norm=1):
    x_batch_list = []
    y_batch_list = []
    w_batch_list = []
    for sample in batch_list:
        x_batch_list.append(sample[0])
        y_batch_list.append(sample[1])
        w_batch_list.append(sample[2])
    x_batch = torch.stack(x_batch_list)
    x_batch = torch.from_numpy(X_scaler.transform(x_batch).astype(np.float32))
    y_batch = torch.stack(y_batch_list)
    w_batch = torch.stack(w_batch_list) / weight_norm
    return x_batch, y_batch, w_batch


def prep_inputs_for_training_mix(batch_list, weight_norm=1):
    x_batch_list = []
    y_batch_list = []
    w_batch_list = []
    for sample in batch_list:
        x_batch_list.append(sample[0])
        y_batch_list.append(sample[1])
        w_batch_list.append(sample[2])
    x_batch = torch.stack(x_batch_list)
    y_batch = torch.stack(y_batch_list)
    w_batch = torch.stack(w_batch_list) / weight_norm
    return x_batch, y_batch, w_batch


def prep_inputs_for_density(batch_list, weight_norm=1):
    x_batch_list = []
    y_batch_list = []
    w_batch_list = []
    for sample in batch_list:
        x_batch_list.append(sample[0])
        y_batch_list.append(sample[1])
        w_batch_list.append(sample[2])
    x_batch = torch.stack(x_batch_list)
    y_batch = torch.stack(y_batch_list)
    w_batch = torch.stack(w_batch_list) / weight_norm
    return x_batch, y_batch, w_batch


def load_scaling(path_to_zip):
    name = osp.splitext(path_to_zip)[0]
    name = os_splitroot(name)[-1]
    with zipfile.ZipFile(path_to_zip, 'r') as zf:
        try:
            scaling_metadata = yaml.load(zf.read("{}_metadata.yaml".format(name)), Loader=yaml.CLoader)["scaling"]
        except KeyError:
            scaling_metadata = yaml.load(zf.read("model_metadata.yaml"), Loader=yaml.CLoader)["scaling"]
    X_scaler = preprocessing.StandardScaler()
    X_scaler.mean_ = scaling_metadata["mean"]
    X_scaler.scale_ = scaling_metadata["scale"]
    X_scaler.var_ = scaling_metadata["var"]
    return X_scaler, scaling_metadata["weights"]