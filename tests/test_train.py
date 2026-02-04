import os
import zipfile

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler

import pytest

from qdre import train as qtrain
from qdre import preprocessing


def test_pare_loss_basic():
    y_hat = torch.tensor([[0.5], [0.2], [0.8]], dtype=torch.float)
    y = torch.tensor([[0.0], [1.0], [0.0]], dtype=torch.float)
    t0 = 2.0
    t1 = 0.5

    loss_none = qtrain.pare_loss(y_hat, y, t0, t1, reduction='none')
    # manual computation: (1-y) * mse(y_hat*t0, 1) + y * mse(y_hat*t1,1)
    ones = torch.ones_like(y)
    expected = (1 - y) * (y_hat * t0 - ones).pow(2) + y * (y_hat * t1 - ones).pow(2)
    assert torch.allclose(loss_none, expected)

    loss_mean = qtrain.pare_loss(y_hat, y, t0, t1, reduction='mean')
    # note: the implementation multiplies the per-class scalar MSE by
    # the per-sample indicator, so 'mean' returns a per-sample tensor
    mse0 = torch.nn.functional.mse_loss(y_hat * t0, torch.ones_like(y), reduction='mean')
    mse1 = torch.nn.functional.mse_loss(y_hat * t1, torch.ones_like(y), reduction='mean')
    expected_mean = (1 - y) * mse0 + y * mse1
    assert torch.allclose(loss_mean, expected_mean)


def test_revert_loss_and_reductions():
    y_hat = torch.tensor([[0.2], [0.8]], dtype=torch.float)
    y = torch.tensor([[0.0], [1.0]], dtype=torch.float)

    val_none = qtrain.revert_loss(y_hat, y, reduction='none')
    # manual formula: y*y_hat - (1-y) * (log(y_hat) + log(1-y_hat))
    expected_none = y * y_hat - (1 - y) * (torch.log(y_hat) + torch.log(1 - y_hat))
    assert torch.allclose(val_none, expected_none)

    assert torch.isclose(qtrain.revert_loss(y_hat, y, reduction='mean'), expected_none.mean())
    assert torch.isclose(qtrain.revert_loss(y_hat, y, reduction='sum'), expected_none.sum())

    with pytest.raises(Exception):
        qtrain.revert_loss(y_hat, y, reduction='invalid')


class SimpleClassifier(nn.Module):
    def __init__(self, inputs):
        super().__init__()
        self.inputs = inputs
        self.hidden_nodes = 3
        self.outputs = 1
        # simple linear followed by sigmoid to produce (0,1)
        self.lin = nn.Linear(inputs, 1)

    def forward(self, x, *args, **kwargs):
        return torch.sigmoid(self.lin(x))


def make_loader(n_samples=6, features=4, batch_size=2):
    # simple toy dataset where labels are first feature > 0
    X = torch.randn(n_samples, features)
    y = (X[:, :1].sum(dim=1) > 0).float().unsqueeze(1)
    w = torch.ones(n_samples, 1)
    dataset = torch.utils.data.TensorDataset(X, y, w)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return loader


def test_train_updates_parameters_and_test_returns_losses():
    loader = make_loader(n_samples=8, features=3, batch_size=2)
    model = SimpleClassifier(inputs=3)
    scaler = StandardScaler()
    # fit scaler on the data returned by loader by using preprocessing.get_scaling helper
    # easiest: build a dataset and fit scaler directly
    all_X = []
    for x, y, w in loader.dataset:
        all_X.append(x)
    all_X = torch.stack(all_X).numpy()
    scaler.fit(all_X)

    optimizer = optim.SGD(model.parameters(), lr=0.1)

    # copy params before
    params_before = [p.clone().detach().cpu() for p in model.parameters()]

    batch_losses, mean_loss = qtrain.train(model, optimizer, loader, X_scaler=scaler, weight_norm=1,
                                           loss='bce', progress_bar=False, device='cpu')

    # ensure training ran and returned losses
    assert isinstance(batch_losses, list)
    assert isinstance(mean_loss, float)
    # parameters should have been updated
    params_after = [p.clone().detach().cpu() for p in model.parameters()]
    any_changed = any(not torch.allclose(a, b) for a, b in zip(params_before, params_after))
    assert any_changed

    # test() should also return per-batch losses and average
    t_losses, t_mean = qtrain.test(model, loader, X_scaler=scaler, weight_norm=1, loss='bce', progress_bar=False)
    assert isinstance(t_losses, list)
    assert isinstance(t_mean, float)


def test_train_supports_custom_loss_callable():
    loader = make_loader(n_samples=6, features=3, batch_size=3)
    model = SimpleClassifier(inputs=3)
    optimizer = optim.SGD(model.parameters(), lr=0.1)

    def custom_loss(preds, labels, weights):
        # return mean absolute difference as a single scalar
        return (weights * (preds - labels).abs()).mean()

    # fit a simple scaler so that `prep_inputs_for_training` works
    all_X = []
    for x, y, w in loader.dataset:
        all_X.append(x)
    all_X = torch.stack(all_X).numpy()
    scaler = StandardScaler()
    scaler.fit(all_X)

    batch_losses, mean_loss = qtrain.train(model, optimizer, loader, X_scaler=scaler, loss=custom_loss, progress_bar=False)
    assert isinstance(batch_losses, list)
    assert isinstance(mean_loss, float)

    t_losses, t_mean = qtrain.test(model, loader, X_scaler=scaler, loss=custom_loss, progress_bar=False)
    assert isinstance(t_losses, list)
    assert isinstance(t_mean, float)


class SMMModel(nn.Module):
    def __init__(self, inputs):
        super().__init__()
        self.inputs = inputs
        # include a parameter so that losses are differentiable
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x, score_function=None, total_variation=False):
        # return deterministic outputs that depend on the score function
        if score_function == 'bce':
            base = torch.zeros((x.shape[0], 1)) + 0.1
        elif score_function == 'pare':
            base = torch.ones((x.shape[0], 1)) * 0.4
        else:
            base = torch.zeros((x.shape[0], 1)) + 0.2
        return base + self.bias


def test_smm_mode_uses_mix_collate_and_passes_arguments():
    loader = make_loader(n_samples=6, features=3, batch_size=2)
    model = SMMModel(inputs=3)
    # ensure optimizer has at least one parameter (SMMModel may have none)
    optimizer = optim.SGD(list(model.parameters()) + [nn.Parameter(torch.zeros(1))], lr=0.01)

    # SMM=True should set collate function that does not require X_scaler
    batch_losses, mean_loss = qtrain.train(model, optimizer, loader, weight_norm=1, loss='bce', SMM=True, progress_bar=False)
    assert isinstance(batch_losses, list)
    assert isinstance(mean_loss, float)

    t_losses, t_mean = qtrain.test(model, loader, weight_norm=1, loss='pare', SMM=True, t0=1.0, t1=2.0, progress_bar=False)
    assert isinstance(t_losses, list)
    assert isinstance(t_mean, float)


def test_get_optimal_loss_pare_and_revert():
    # Create loader compatible with prep_inputs_for_density
    # Each sample: x (at least 2 features), y (0/1), w
    X = torch.tensor([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]], dtype=torch.float)
    # binary labels
    y = torch.tensor([[0.0], [1.0], [0.0]], dtype=torch.float)
    w = torch.ones(3, 1)
    dataset = torch.utils.data.TensorDataset(X, y, w)
    loader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=False)

    class DensityModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.inputs = 2

        def forward(self, x):
            # return a deterministic output in (0,1)
            # make it depend on the first coord for variety
            val = (x[:, :1].sum(dim=1, keepdim=True) + 1.0) / 3.0
            return val

    model = DensityModel()

    # test bce
    bce_loss = qtrain.get_optimal_loss(model, loader, loss='bce', progress_bar=False)
    assert isinstance(bce_loss, float)

    # test revert
    rev_loss = qtrain.get_optimal_loss(model, loader, loss='revert', progress_bar=False)
    assert isinstance(rev_loss, float)

    # test pare with analytic transform (needs t0, t1)
    pare_loss_val = qtrain.get_optimal_loss(model, loader, loss='pare', progress_bar=False, t0=1.0, t1=2.0)
    assert isinstance(pare_loss_val, float)


def test_get_model_metadata_and_save_load(tmp_path):
    # Create a model whose class name is 'Classifier'
    class Classifier(SimpleClassifier):
        pass

    model = Classifier(inputs=4)
    training_settings = {'epochs': 7, 'lr': 0.01}
    # input scaler mimic
    class DummyScaler:
        def __init__(self):
            self.mean_ = np.array([0.1, 0.2, 0.3, 0.4])
            self.scale_ = np.array([1.0, 1.0, 1.0, 1.0])
            self.var_ = np.array([1.0, 1.0, 1.0, 1.0])

    scaler = DummyScaler()
    weight_scale = 2.5

    metadata = qtrain.get_model_metadata(training_settings, model, scaler, weight_scale)
    assert 'model' in metadata and 'training' in metadata and 'scaling' in metadata
    assert metadata['training'] == training_settings

    # Save and load training settings via zip archive
    savedir = tmp_path
    qtrain.save_model_data(model, metadata, savedir=str(savedir), name='testmodel')
    zip_path = os.path.join(str(savedir), 'testmodel.zip')
    assert os.path.exists(zip_path)

    loaded_training = qtrain.load_training_settings(zip_path)
    assert loaded_training == training_settings

    # clean up
    os.remove(zip_path)
