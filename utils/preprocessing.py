"""Data preprocessing helpers.

This module provides light-weight utilities for loading NumPy-based
datasets, converting them to PyTorch `TensorDataset` objects, building
combined datasets, computing input scaling (StandardScaler), and
preparing batched inputs for training and evaluation loops used in the
project.

Functions follow the convention that a data loader yields tuples of
`(x, y, w)` where `x` is a feature tensor, `y` is the label tensor and
`w` is a one-column weight tensor.
"""

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
        """Light-weight wrapper for a single NumPy data file.

        The class expects files created as NumPy arrays where the last
        column represents per-event weights and the remaining columns
        represent flattened input features.

        Args:
            file_name (str): path to the .npy file.
            label (int|float): class label to assign to all examples in
                the file (0/1 for binary classification).
            start_event (int): index of the first data point to
                include (inclusive).
            stop_event (int|None): index of the last data point to
                include (exclusive). If None, the file's full length is
                used.
        """
        self.file_name = file_name
        self.label = label
        self.start_event = start_event
        self.stop_event = stop_event
        self.datas = []


    def process(self, normalize_weights=False, weight_norm=None):
        """Load the NumPy file and convert it to a `TensorDataset`.

        The method reads `self.file_name` using `np.load`, slices rows
        between `start_event` and `stop_event`, and splits columns into
        features `x` (all columns except last) and weights `w` (last
        column). Labels `y` are constructed from `self.label`.

        If `normalize_weights` is True the returned `weight_norm` can be
        used to rescale weights so that their average becomes 1.

        Returns:
            float|None: computed `weight_norm` if normalization is used,
            otherwise the passed-in `weight_norm` (possibly None).
        """
        self.datas = []
        # Load binary NumPy array from file
        with open(self.file_name, 'rb') as f:
            data = np.load(f)

        if self.stop_event is None:
            self.stop_event = int(data.shape[0])
        self.n_events = self.stop_event - self.start_event

        # Features are all but the last column; weights are the last
        x = torch.tensor(
            data[self.start_event:self.stop_event, :-1].reshape(self.n_events, -1),
            dtype=torch.float)
        w = torch.tensor(
            data[self.start_event:self.stop_event, -1].reshape(self.n_events, 1),
            dtype=torch.float)

        if normalize_weights is True:
            if weight_norm is None:
                weight_norm = w.mean()
            w = w / weight_norm

        # Construct label tensor: supports binary or arbitrary scalar label
        if self.label == 0:
            y = torch.zeros((self.n_events, 1), dtype=torch.float)
        elif self.label == 1:
            y = torch.ones((self.n_events, 1), dtype=torch.float)
        else:
            y = torch.ones((self.n_events, 1), dtype=torch.float) * self.label

        self.datas.append(TensorDataset(x, y, w))
        return weight_norm


def CombinedDataset(*args):
    """Concatenate multiple `Dataset` wrappers into a single PyTorch
    `ConcatDataset`.

    Each argument is expected to be an instance of the `Dataset` class
    above (or any object exposing a `.datas` attribute containing
    `TensorDataset` objects). The function returns a `ConcatDataset`
    ready to be consumed by a `DataLoader`.
    """
    total = []
    for arg in args:
        total.extend(arg.datas)
    return ConcatDataset(total)


def get_scaling(dataset, batch_size=1024, shuffle=False):
    """Compute input scaling (StandardScaler) and average weight.

    The function iterates over `dataset` with a DataLoader that uses
    `prep_inputs_for_scaling` to collect `x` and `w`. It performs
    incremental `partial_fit` calls on a `StandardScaler` instance to
    avoid loading all data into memory at once.

    Returns:
        (StandardScaler, float): fitted scaler and average event
        weight computed across the dataset.
    """
    scaling_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    scaling_loader.collate_fn = prep_inputs_for_scaling

    X_scaler = preprocessing.StandardScaler()
    weight_total = 0
    num_total_weights = 0

    for batch in tqdm(scaling_loader):
        # batch is (x_batch, w_batch)
        X_scaler.partial_fit(batch[0])
        weight_total += batch[1].sum()
        num_total_weights += len(batch[1])
    weight_norm = weight_total / num_total_weights
    return X_scaler, weight_norm


def prep_inputs_for_scaling(batch_list):
    """Collate function for `get_scaling` DataLoader.

    The function expects `batch_list` to be a list of `TensorDataset`
    entries (x, y, w) and returns a tuple `(x_batch, w_batch)` where
    `x_batch` stacks the feature tensors and `w_batch` stacks the
    corresponding weight tensors.
    """
    x_batch_list = []
    w_batch_list = []
    for sample in batch_list:
        x_batch_list.append(sample[0])
        w_batch_list.append(sample[2])
    x_batch = torch.stack(x_batch_list)
    w_batch = torch.stack(w_batch_list)
    return x_batch, w_batch


def prep_inputs_for_training(batch_list, X_scaler, weight_norm=1):
    """Collate function for training DataLoader that applies scaling.

    Args:
        batch_list (list): list of `(x, y, w)` samples.
        X_scaler (StandardScaler): fitted scaler used to transform
            input features. The scaler expects a NumPy array input so
            tensors are converted accordingly.
        weight_norm (float): value used to normalise weights (divide
            raw weights by this number).

    Returns:
        (Tensor, Tensor, Tensor): `(x_batch, y_batch, w_batch)` ready
        to be moved to the training device.
    """
    x_batch_list = []
    y_batch_list = []
    w_batch_list = []
    for sample in batch_list:
        x_batch_list.append(sample[0])
        y_batch_list.append(sample[1])
        w_batch_list.append(sample[2])
    x_batch = torch.stack(x_batch_list)
    # sklearn scaler operates on NumPy arrays; convert and cast to float32
    x_batch = torch.from_numpy(X_scaler.transform(x_batch).astype(np.float32))
    y_batch = torch.stack(y_batch_list)
    w_batch = torch.stack(w_batch_list) / weight_norm
    return x_batch, y_batch, w_batch


def prep_inputs_for_training_mix(batch_list, weight_norm=1):
    """Collate function for mixture-model training.

    This behaves like `prep_inputs_for_training` but does not apply an
    external scaler to the inputs (mixture models expect raw inputs).
    """
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
    """Collate function for density-evaluation DataLoader.

    Same as `prep_inputs_for_training_mix` but kept as a separate
    function for clarity and future extensions specific to density
    estimation pipelines.
    """
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
    """Load scaler parameters from a model archive zip.

    The function looks for a `<name>_metadata.yaml` entry inside the
    provided zip and reads the 'scaling' section to reconstruct a
    `StandardScaler` instance with matching `mean_`, `scale_` and
    `var_` attributes. It also returns the saved weight normalization
    constant.

    Args:
        path_to_zip (str): path to a zip file created by
            `save_model_data` containing the serialized metadata.

    Returns:
        (StandardScaler, float): reconstructed scaler and stored
        weight normalization value.
    """
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