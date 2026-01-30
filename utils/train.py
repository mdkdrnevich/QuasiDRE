"""Training utilities: loss helpers, training and evaluation loops,
and model serialization helpers.

This module provides small convenience wrappers used across training
scripts: special loss definitions, `train` and `test` loops that handle
data collation, mixture models (SMM) as a special case, and functions
to save model metadata and parameters.
"""

import torch
import torch.nn.functional as F
import numpy as np

import zipfile
import yaml
import os
import os.path as osp
import collections
from tqdm import tqdm
import types

from . import preprocessing as carl_preprocessing
from . import models as carl_models



def pare_loss(y_hat, y, t0, t1, reduction='none'):
    """Pare loss used by mixture classifiers.

    The Pare loss is a weighted combination of mean-squared-error terms
    where the predicted score `y_hat` is rescaled by different
    temperature-like constants `t0` and `t1` depending on the true
    label. For label y=0 the prediction is compared to 1 after scaling
    by `t0`, and for y=1 it is compared to 1 after scaling by `t1`.

    Args:
        y_hat (Tensor): model output, expected in the same shape as `y`.
        y (Tensor): binary labels (0 or 1).
        t0 (float|Tensor): scaling applied for negative examples.
        t1 (float|Tensor): scaling applied for positive examples.
        reduction (str): one of 'none', 'mean', or 'sum' forwarded to
            `torch.nn.functional.mse_loss`.

    Returns:
        Tensor: loss tensor with the requested reduction.
    """
    # Target vector of ones for the MSE comparisons
    ones = torch.ones_like(y)
    # Compute MSE separately for y==0 and y==1 paths and combine
    return (1 - y) * F.mse_loss(y_hat * t0, ones, reduction=reduction) + \
           y * F.mse_loss(y_hat * t1, ones, reduction=reduction)


def qdre_loss(y_hat, y, reduction='none'):
    """Quasi-DRE (QDRE) loss used for density-ratio estimation.

    The implementation follows the form used in the project: for
    positives the loss contribution grows with `y_hat` linearly, while
    for negatives it uses the negative log-probabilities.

    WARNING: `y_hat` values must be strictly inside (0, 1) to avoid
    log(0) issues.

    Args:
        y_hat (Tensor): model score/probability in (0,1).
        y (Tensor): binary labels (0 or 1).
        reduction (str): 'none', 'mean', or 'sum'.

    Returns:
        Tensor: elementwise or reduced loss value.
    """
    rval = y * y_hat - (1 - y) * (torch.log(y_hat) + torch.log(1 - y_hat))
    if reduction == 'mean':
        return torch.mean(rval)
    elif reduction == 'sum':
        return torch.sum(rval)
    elif reduction == 'none':
        return rval
    else:
        raise Exception("Reduction not implemented: ", reduction)
    

@torch.no_grad()
def test(model, loader, X_scaler=None, weight_norm=1, loss='bce', max_num_batches=np.inf,
         SMM=False, total_variation=False, progress_bar=True, leave=False, device='cpu', **kwargs):
    """Evaluate `model` over `loader` without updating parameters.

    This function performs a forward pass over the dataset provided by
    `loader` and returns a list of per-batch loss values plus the mean
    loss. The function supports standard binary cross-entropy ('bce'),
    mean-squared-error ('mse'), the custom Pare loss ('pare'), the
    QDRE loss ('qdre'), or an arbitrary callable `loss` that accepts
    `(predictions, labels, weights)` and returns a Tensor.

    Special handling for SMM (mixture) models: when `SMM=True` the
    loader's collate function prepares inputs for mixture training and
    the model is called with `score_function` and
    `total_variation` flags.

    Args:
        model (torch.nn.Module): model to evaluate.
        loader (DataLoader): yields batches of `(x, y, w)`.
        X_scaler: optional input scaler passed to preprocessing.
        weight_norm (float): normalization factor for preprocessing.
        loss (str|callable): which loss to compute or a callable.
        max_num_batches (int): early-stop after this many batches.
        SMM (bool): whether model is a mixture/SMM model.
        total_variation (bool): forwarded to model when SMM is True.
        progress_bar (bool): whether to show tqdm progress.
        leave (bool): tqdm `leave` flag.
        device (str|torch.device): device to move tensors to.
        **kwargs: extra arguments (e.g., 't0' and 't1' for Pare loss).

    Returns:
        (list, float): tuple of per-batch loss list and average loss.
    """
    # Configure loader collate function depending on whether we are
    # dealing with a mixture (SMM) or a standard classifier.
    if SMM is True:
        loader.collate_fn = lambda batch: carl_preprocessing.prep_inputs_for_training_mix(
            batch, weight_norm=weight_norm)
    elif SMM is False:
        loader.collate_fn = lambda batch: carl_preprocessing.prep_inputs_for_training(
            batch, X_scaler, weight_norm=weight_norm)

    # Pare loss requires temperature parameters
    if loss == "pare":
        t0 = kwargs.get("t0")
        t1 = kwargs.get("t1")

    model.eval()

    batch_losses = []

    sum_loss = 0.0
    # Choose iterator with or without a progress bar
    if progress_bar is True:
        t = tqdm(enumerate(loader), total=len(loader), position=0, leave=leave)
    else:
        t = enumerate(loader)

    for i, batch in t:
        if i + 1 >= max_num_batches:
            break
        # Expect batch to be (x, y, w)
        x = batch[0].to(device)
        y = batch[1].to(device)
        w = batch[2].to(device)

        # Get model output; mixture models may expose a different
        # calling convention (score_function, total_variation).
        if SMM is False:
            batch_output = model(x[:, :model.inputs])
        elif SMM is True:
            batch_output = model(x[:, :model.inputs], score_function=loss,
                                 total_variation=total_variation)

        # Compute batch loss according to chosen loss function
        if loss == 'bce':
            batch_loss_item = F.binary_cross_entropy(batch_output, y, weight=w).cpu().item()
        elif loss == 'mse':
            batch_loss_item = (F.mse_loss(batch_output, y, reduction='none') * w).mean().cpu().item()
        elif loss == "pare":
            batch_loss_item = (pare_loss(batch_output, y, t0, t1, reduction='none') * w).mean().cpu().item()
        elif loss == "qdre":
            batch_loss_item = (qdre_loss(batch_output, y, reduction='none') * w).mean().cpu().item()
        elif type(loss) is types.FunctionType:
            # custom loss signature: (preds, labels, weights)
            batch_loss_item = loss(batch_output, y, w).cpu().item()
        else:
            raise Exception("Loss not implemented")

        batch_losses.append(batch_loss_item)
        sum_loss += batch_loss_item
        if progress_bar is True:
            t.set_description("loss = %.5f" % (batch_loss_item))
            t.refresh()  # to show immediately the update

    # Return per-batch numbers and the mean over processed batches
    return batch_losses, sum_loss / (i + 1)



def train(model, optimizer, loader, X_scaler=None, weight_norm=1, loss='bce', max_num_batches=np.inf,
          SMM=False, total_variation=False, progress_bar=True, leave=False, device='cpu', **kwargs):
    """Train `model` for one epoch (or up to `max_num_batches`).

    This function runs a training loop: for each batch it computes the
    selected loss, calls `backward()`, and advances the optimizer.
    The interface mirrors `test()` with the addition of `optimizer` and
    calls to `.train()` and `.step()`.

    Args:
        model (torch.nn.Module): model to train.
        optimizer (torch.optim.Optimizer): optimizer to step.
        loader (DataLoader): yields batches of `(x, y, w)`.
        X_scaler: optional scaler forwarded to preprocessing.
        weight_norm (float): normalization factor for preprocessing.
        loss (str|callable): loss choice or callable.
        max_num_batches (int): stop early after this many batches.
        SMM (bool): handle mixture model calling convention.
        total_variation (bool): forwarded to model when SMM is True.
        progress_bar (bool), leave (bool), device: as in `test()`.
        **kwargs: extra options (e.g., 't0'/'t1' for Pare loss).

    Returns:
        (list, float): per-batch losses and the mean loss.
    """
    # Configure loader collate function for training
    if SMM is True:
        loader.collate_fn = lambda batch: carl_preprocessing.prep_inputs_for_training_mix(
            batch, weight_norm=weight_norm)
    elif SMM is False:
        loader.collate_fn = lambda batch: carl_preprocessing.prep_inputs_for_training(
            batch, X_scaler, weight_norm=weight_norm)

    if loss == "pare":
        t0 = kwargs.get("t0")
        t1 = kwargs.get("t1")

    model.train()

    batch_losses = []

    sum_loss = 0.0
    if progress_bar is True:
        t = tqdm(enumerate(loader), total=len(loader), position=0, leave=leave)
    else:
        t = enumerate(loader)

    for i, batch in t:
        if i + 1 >= max_num_batches:
            break
        optimizer.zero_grad()
        x = batch[0].to(device)
        y = batch[1].to(device)
        w = batch[2].to(device)

        # Forward pass: take only the input columns used by the model
        if SMM is False:
            batch_output = model(x[:, :model.inputs])
        elif SMM is True:
            batch_output = model(x[:, :model.inputs], score_function=loss,
                                 total_variation=total_variation)

        # Compute the loss according to the requested objective
        if loss == 'bce':
            batch_loss = F.binary_cross_entropy(batch_output, y, weight=w)
        elif loss == 'mse':
            batch_loss = (F.mse_loss(batch_output, y, reduction='none') * w).mean()
        elif loss == "pare":
            batch_loss = (pare_loss(batch_output, y, t0, t1, reduction='none') * w).mean()
        elif loss == "qdre":
            batch_loss = (qdre_loss(batch_output, y, reduction='none') * w).mean()
        elif type(loss) is types.FunctionType:
            batch_loss = loss(batch_output, y, w)
        else:
            raise Exception("Loss not implemented")

        # Backprop and optimizer step
        batch_loss.backward()
        batch_loss_item = batch_loss.item()
        batch_losses.append(batch_loss_item)
        if progress_bar is True:
            t.set_description("loss = %.5f" % batch_loss_item)
            t.refresh()
        sum_loss += batch_loss_item
        optimizer.step()

    return batch_losses, sum_loss / (i + 1)




def get_optimal_loss(model, loader, weight_norm=1, loss='bce',
                     progress_bar=True, leave=False, device='cpu', **kwargs):
    """Estimate the optimal (post-processed) loss for a density model.

    This helper is used when evaluating a model that predicts an
    intermediate score (e.g. density estimates) and a post-processing
    step is required to convert the raw output into a final prediction
    before computing the loss. The routine assumes `loader` yields
    batches prepared by `prep_inputs_for_density`.

    Special-case for the Pare loss: the function computes an analytic
    re-scaling `batch_s` from the model score before applying the
    `pare_loss`.

    Args:
        model (torch.nn.Module): should accept inputs where the first
            two columns are used (hence `x[:, :2]` below).
        loader (DataLoader): yields `(x, y, w)`.
        weight_norm (float): normalization forwarded to preprocessing.
        loss (str): 'bce', 'mse', 'pare', or 'qdre'.
        progress_bar, leave, device: display and device controls.
        **kwargs: extra parameters for loss computations (e.g., t0, t1).

    Returns:
        float: mean loss across processed batches.
    """
    loader.collate_fn = lambda batch: carl_preprocessing.prep_inputs_for_density(
        batch, weight_norm=weight_norm)

    sum_loss = 0.0
    if progress_bar is True:
        t = tqdm(enumerate(loader), total=len(loader), position=0, leave=leave)
    else:
        t = enumerate(loader)

    for i, batch in t:
        # Note: inputs are kept on CPU for the model call below, then
        # results are moved to the desired device.
        x = batch[0].to('cpu')
        y = batch[1].to(device)
        w = batch[2].to(device)
        batch_output = model(x[:, :2]).reshape(-1, 1).to(device)

        if loss == 'bce':
            batch_loss_item = F.binary_cross_entropy(batch_output, y, weight=w).cpu().item()
        elif loss == 'mse':
            batch_loss_item = (F.mse_loss(batch_output, y, reduction='none') * w).mean().cpu().item()
        elif loss == 'pare':
            # Recover the optimal transformed score for Pare objective
            t0 = kwargs.get('t0')
            t1 = kwargs.get('t1')
            batch_r = batch_output / (1 - batch_output)
            batch_s = (t0 + t1 * batch_r) / (t0**2 + t1**2 * batch_r)
            batch_loss_item = (pare_loss(batch_s, y, t0, t1, reduction='none') * w).mean().cpu().item()
        elif loss == 'qdre':
            batch_loss_item = (qdre_loss(batch_output, y, reduction='none') * w).mean().cpu().item()

        sum_loss += batch_loss_item
        if progress_bar is True:
            t.set_description("loss = %.5f" % (batch_loss_item))
            t.refresh()

    return sum_loss / (i + 1)


def get_model_metadata(training_settings, model, input_scaler, weight_scale):
    """Compose a serializable metadata dict for the given `model`.

    The returned dictionary contains three top-level keys: 'model'
    (architecture and hyperparameters), 'training' (user-supplied
    training settings), and 'scaling' (input/output scaler parameters
    and weight normalisation).

    Args:
        training_settings (dict): settings used for training (saved as-is).
        model (torch.nn.Module): trained model instance. The function
            inspects the class name and extracts known attributes for
            supported model types.
        input_scaler: scaler object exposing `mean_`, `scale_`, and
            `var_` attributes (e.g., sklearn scaler).
        weight_scale: numeric weight-scaling used during preprocessing.

    Returns:
        dict: metadata suitable for YAML serialization.
    """
    classname = model.__class__.__name__
    model_settings = {"name": classname}
    STANDARD_CLASSIFIERS = ["Classifier", "Regression"]
    MIXTURE_CLASSIFIERS = ["MixtureClassifier", "SingleMixtureClassifier"]

    # Extract model-specific attributes based on class name
    if classname in STANDARD_CLASSIFIERS:
        model_settings.update({
            "inputs": model.inputs,
            "hidden_nodes": model.hidden_nodes,
            "outputs": model.outputs
        })
    elif classname in MIXTURE_CLASSIFIERS:
        model_settings.update({
            "subclassifier_paths": model.subclassifier_paths,
            "c0": model.c0,
            "c1": model.c1,
            "t0": model.t0,
            "t1": model.t1,
            "fine_tune": model.fine_tune
        })
        if classname == "SingleMixtureClassifier":
            model_settings.update({
                "which_mixture": model._which_mixture,
            })
    else:
        raise Exception("Not implemented for class:", classname)

    # Save scaler parameters for reproducing preprocessing at inference
    scaling_settings = {
        "mean": input_scaler.mean_,
        "scale": input_scaler.scale_,
        "var": input_scaler.var_,
        "weights": weight_scale
    }
    # Regression models may expose output scaling information
    if classname in ["Regression",]:
        output_mean, output_scale = model.get_output_scaling()
        scaling_settings.update({
            "output_mean": output_mean,
            "output_std": output_scale
        })

    metadata = {
        "model": model_settings,
        "training": training_settings,
        "scaling": scaling_settings
    }
    return metadata


def save_model_data(model, metadata, savedir='.', name="model", save_onnx=True, device='cpu'):
    """Persist model parameters and metadata into a zip archive.

    The function writes a temporary YAML file with `metadata`, the
    PyTorch state dict and bundles them into `<name>.zip`. Temporary
    intermediate files are removed before returning.

    Args:
        model (torch.nn.Module): model whose state_dict will be saved.
        metadata (dict): metadata to serialize to YAML.
        name (str): base name for created files (defaults to 'model').
        save_onnx (bool): unused in practice (ONNX export block is
            currently commented out) but kept for backward compatibility.
        device (str|torch.device): device to move model to when
            exporting (unused while ONNX export is commented).

    Returns:
        None
    """
    yaml_name = "{}_metadata.yaml".format(name)
    pth_name = "{}.pth".format(name)
    yaml_path = osp.join(savedir, yaml_name)
    pth_path = osp.join(savedir, pth_name)
    zip_path = osp.join(savedir, "{}.zip".format(name))
    yaml.dump(metadata, open(yaml_path, 'w'))
    torch.save(model.state_dict(), pth_path)
    """
    if save_onnx is True:
        model = model.to('cpu')
        model.eval()
        test_input = []
        input_names = []
        dynamic_axes = {}
        for k in model.features:
            test_input.append(torch.randn(2, model.features[k]["size"]).T[None, :])
            input_names.append(k)
            if model.features[k]["set"] is True:
                dynamic_axes[k] = {2 : "batch_and_set_size"}
            else:
                dynamic_axes[k] = {1 : "batch_size"}
        #test_input.append(torch.ones(len(model.features), 1, dtype=int))
        test_input.append(torch.cat([torch.arange(2)[None,:] for _ in range(len(model.features))], dim=0))
        input_names.append("sample_indices")
        dynamic_axes["sample_indices"] = {1 : "batch_size"}
        dynamic_axes["output"] = {0 : "batch_size"}
        test_input = tuple(test_input)

        traced_model = torch.jit.trace(model, example_inputs=test_input)
        torch.onnx.export(traced_model,
                          test_input,
                          "deepsets_model.onnx",
                          export_params=True,
                          opset_version=16,
                          do_constant_folding=True,
                          input_names=input_names,
                          output_names=["output"],
                          dynamic_axes=dynamic_axes)
        model = model.to(device)
    """
        
    # Bundle files into a single archive for convenient distribution
    with zipfile.ZipFile(zip_path, mode='w') as zipf:
        zipf.write(yaml_path, arcname=yaml_name)
        zipf.write(pth_path, arcname=pth_name)

    # Clean up temporary files
    os.remove(yaml_path)
    os.remove(pth_path)
    return None


def load_training_settings(path_to_zip):
    """Load and return the `training` section from a model archive.

    The archive is expected to contain a YAML file named either
    `<name>_metadata.yaml` (preferred) or `model_metadata.yaml`. The
    function returns the parsed value under the 'training' key.

    Args:
        path_to_zip (str): path to the `<name>.zip` produced by
            `save_model_data`.

    Returns:
        dict: training settings previously stored in metadata.
    """
    name = osp.split(osp.splitext(path_to_zip)[0])[-1]
    with zipfile.ZipFile(path_to_zip, 'r') as zf:
        try:
            return yaml.load(zf.read("{}_metadata.yaml".format(name)), Loader=yaml.CLoader)["training"]
        except KeyError:
            return yaml.load(zf.read("model_metadata.yaml"), Loader=yaml.CLoader)["training"]