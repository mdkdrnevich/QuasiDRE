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

from .tools import os_splitroot
from . import preprocessing as carl_preprocessing
from . import models as carl_models



def pare_loss(y_hat, y, t0, t1, reduction='none'):
    ones = torch.ones_like(y)
    return (1 - y)*F.mse_loss(y_hat*t0, ones, reduction=reduction) + y*F.mse_loss(y_hat*t1, ones, reduction=reduction)


def qdre_loss(y_hat, y, reduction='none'):
    rval = y*y_hat - (1-y)*(torch.log(y_hat) + torch.log(1-y_hat))
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
    if SMM is True:
        loader.collate_fn = lambda batch: carl_preprocessing.prep_inputs_for_training_mix(batch, weight_norm=weight_norm)
    elif SMM is False:
        loader.collate_fn = lambda batch: carl_preprocessing.prep_inputs_for_training(batch, X_scaler, weight_norm=weight_norm)

    if loss == "pare":
        t0 = kwargs.get("t0")
        t1 = kwargs.get("t1")

    model.eval()

    batch_losses = []

    sum_loss = 0.0
    if progress_bar is True:
        t = tqdm(enumerate(loader), total=len(loader), position=0, leave=leave)
    else:
        t = enumerate(loader)
    for i, batch in t:
        if i+1 >= max_num_batches:
            break
        x = batch[0].to(device)
        y = batch[1].to(device)
        w = batch[2].to(device)

        # Get the score output depending on the model and loss
        if SMM is False:
            batch_output = model(x[:,:model.inputs])
        elif SMM is True:
            batch_output = model(x[:,:model.inputs], score_function=loss, total_variation=total_variation)

        # Evaluate the loss
        if loss == 'bce':
            batch_loss_item = F.binary_cross_entropy(batch_output, y, weight=w).cpu().item()
        elif loss == 'mse':
            batch_loss_item = (F.mse_loss(batch_output, y, reduction='none')*w).mean().cpu().item()
        elif loss == "pare":
            batch_loss_item = (pare_loss(batch_output, y, t0, t1, reduction='none')*w).mean().cpu().item()
        elif loss == "qdre":
            batch_loss_item = (qdre_loss(batch_output, y, reduction='none')*w).mean().cpu().item()
        elif type(loss) is types.FunctionType:
            batch_loss_item = loss(batch_output, y, w).cpu().item()
        else:
            raise Exception("Loss not implemented")

        batch_losses.append(batch_loss_item)
        sum_loss += batch_loss_item
        if progress_bar is True:
            t.set_description("loss = %.5f" % (batch_loss_item))
            t.refresh()  # to show immediately the update

    return batch_losses, sum_loss / (i + 1)



def train(model, optimizer, loader, X_scaler=None, weight_norm=1, loss='bce', max_num_batches=np.inf,
          SMM=False, total_variation=False, progress_bar=True, leave=False, device='cpu', **kwargs):
    if SMM is True:
        loader.collate_fn = lambda batch: carl_preprocessing.prep_inputs_for_training_mix(batch, weight_norm=weight_norm)
    elif SMM is False:
        loader.collate_fn = lambda batch: carl_preprocessing.prep_inputs_for_training(batch, X_scaler, weight_norm=weight_norm)

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
        #print("BATCH NUMBER:", i)
        if i+1 >= max_num_batches:
            break
        optimizer.zero_grad()
        x = batch[0].to(device)
        y = batch[1].to(device)
        w = batch[2].to(device)

        # Get the score output depending on the model and loss
        if SMM is False:
            batch_output = model(x[:,:model.inputs])
        elif SMM is True:
            batch_output = model(x[:,:model.inputs], score_function=loss, total_variation=total_variation)
            
        if loss == 'bce':
            batch_loss = F.binary_cross_entropy(batch_output, y, weight=w)
        elif loss == 'mse':
            batch_loss = (F.mse_loss(batch_output, y, reduction='none')*w).mean()
        elif loss == "pare":
            batch_loss = (pare_loss(batch_output, y, t0, t1, reduction='none')*w).mean()
        elif loss == "qdre":
            batch_loss = (qdre_loss(batch_output, y, reduction='none')*w).mean()
        elif type(loss) is types.FunctionType:
            batch_loss = loss(batch_output, y, w)
        else:
            raise Exception("Loss not implemented")

        batch_loss.backward()
        batch_loss_item = batch_loss.item()
        batch_losses.append(batch_loss_item)
        if progress_bar is True:
            t.set_description("loss = %.5f" % batch_loss_item)
            t.refresh()  # to show immediately the update
        sum_loss += batch_loss_item
        optimizer.step()

    return batch_losses, sum_loss / (i + 1)




def get_optimal_loss(model, loader, weight_norm=1, loss='bce',
                     progress_bar=True, leave=False, device='cpu', **kwargs):
    loader.collate_fn = lambda batch: carl_preprocessing.prep_inputs_for_density(batch, weight_norm=weight_norm)

    sum_loss = 0.0
    if progress_bar is True:
        t = tqdm(enumerate(loader), total=len(loader), position=0, leave=leave)
    else:
        t = enumerate(loader)
    for i, batch in t:
        x = batch[0].to('cpu')
        y = batch[1].to(device)
        w = batch[2].to(device)
        batch_output = model(x[:,:2]).reshape(-1,1).to(device)
        if loss == 'bce':
            batch_loss_item = F.binary_cross_entropy(batch_output, y, weight=w).cpu().item()
        elif loss == 'mse':
            batch_loss_item = (F.mse_loss(batch_output, y, reduction='none')*w).mean().cpu().item()
        elif loss == 'pare':
            t0 = kwargs.get('t0')
            t1 = kwargs.get('t1')
            batch_r = batch_output / (1 - batch_output)
            batch_s = (t0 + t1 * batch_r) / (t0**2 + t1**2 * batch_r)
            batch_loss_item = (pare_loss(batch_s, y, t0, t1, reduction='none')*w).mean().cpu().item()
        elif loss == 'qdre':
            batch_loss_item = (qdre_loss(batch_output, y, reduction='none')*w).mean().cpu().item()
        sum_loss += batch_loss_item
        if progress_bar is True:
            t.set_description("loss = %.5f" % (batch_loss_item))
            t.refresh()  # to show immediately the update

    return sum_loss / (i + 1)


def get_model_metadata(training_settings, model, input_scaler, weight_scale):
    classname = model.__class__.__name__
    model_settings = {"name": classname}
    STANDARD_CLASSIFIERS = ["Classifier", "Regression"]
    MIXTURE_CLASSIFIERS = ["MixtureClassifier", "SingleMixtureClassifier"]
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
    
    scaling_settings = {
        "mean": input_scaler.mean_,
        "scale": input_scaler.scale_,
        "var": input_scaler.var_,
        "weights": weight_scale
    }
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


def save_model_data(model, metadata, name="model", save_onnx=True, device='cpu'):
    yaml.dump(metadata, open("{}_metadata.yaml".format(name), 'w'))
    torch.save(model.state_dict(), "{}.pth".format(name))
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
        
    with zipfile.ZipFile("{}.zip".format(name), mode='w') as zipf:
        zipf.write("{}_metadata.yaml".format(name))
        zipf.write("{}.pth".format(name))
        #zipf.write("model.onnx")

    os.remove("{}_metadata.yaml".format(name))
    os.remove("{}.pth".format(name))
    #os.remove("model.onnx")
    return None


def load_training_settings(path_to_zip):
    name = osp.splitext(path_to_zip)[0]
    name = os_splitroot(name)[-1]
    with zipfile.ZipFile(path_to_zip, 'r') as zf:
        try:
            return yaml.load(zf.read("{}_metadata.yaml".format(name)), Loader=yaml.CLoader)["training"]
        except KeyError:
            return yaml.load(zf.read("model_metadata.yaml"), Loader=yaml.CLoader)["training"]