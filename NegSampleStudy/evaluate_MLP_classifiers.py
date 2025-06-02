#!/usr/bin/env python
# coding: utf-8

import os.path as osp
import sys
sys.path.append('../')
import torch
from torch.utils.data import Dataset, TensorDataset, DataLoader
import torch.nn.functional as F

import multiprocessing as mp
import time
import os
import shutil
import queue

import numpy as np
from sklearn import preprocessing, metrics
import types


import utils
import utils.Camel.equations as equations

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#DEVICE = 'cpu'
print(DEVICE)


from utils import preprocessing as carl_preprocessing
from utils import models as carl_models


@torch.no_grad()
def get_scores(model, loader, X_scaler=None, weight_norm=1, mix=False, leave=False, calibrator=None):
    if type(model) is not types.FunctionType:
        model.eval()
        if mix is True:
            loader.collate_fn = lambda batch: utils.preprocessing.prep_inputs_for_training_mix(batch, weight_norm=weight_norm)
        else:
            loader.collate_fn = lambda batch: utils.preprocessing.prep_inputs_for_training(batch, X_scaler, weight_norm=weight_norm)
    else:
        loader.collate_fn = lambda batch: utils.preprocessing.prep_inputs_for_density(batch, weight_norm=weight_norm)

    score_list = []
    target_list = []
    weight_list = []
    for i, batch in enumerate(loader):
        target_list.append(batch[1])
        weight_list.append(batch[2])
        if type(model) is not types.FunctionType:
            x = batch[0].to(DEVICE)
        else:
            x = batch[0].to('cpu')
        batch_score = model(x)
        if calibrator is not None:
            batch_score = calibrator.predict(batch_score.cpu())
        score_list.append(batch_score)

    if calibrator is None:
        score_list = torch.cat(score_list).cpu().numpy().flatten()
    else:
        score_list = np.concatenate(score_list).flatten()
    return score_list, torch.cat(target_list).cpu().numpy().flatten(), torch.cat(weight_list).cpu().numpy().flatten()


@torch.no_grad()
def get_r_hats(model, loader, X_scaler=None, weight_norm=1, mix=False, leave=False, classifier="s", labels=None):
    if type(model) is not types.FunctionType:
        model.eval()
        if mix is True:
            loader.collate_fn = lambda batch: utils.preprocessing.prep_inputs_for_training_mix(batch, weight_norm=weight_norm)
        else:
            loader.collate_fn = lambda batch: utils.preprocessing.prep_inputs_for_training(batch, X_scaler, weight_norm=weight_norm)
    else:
        loader.collate_fn = lambda batch: utils.preprocessing.prep_inputs_for_density(batch, weight_norm=weight_norm)

    r_hat_list = []
    for i, batch in enumerate(loader):
        if type(model) is not types.FunctionType:
            x = batch[0].to(DEVICE)
        else:
            x = batch[0].to('cpu')
        batch_output = model(x)
        if classifier == "s":
            r_hat = batch_output / (1 - batch_output)
        elif classifier == "h":
            y0, y1 = labels
            r_hat = -y0*(1-y0*batch_output)/(y1*(1-y1*batch_output))
        elif classifier == 'new':
            r_hat = (1-2*batch_output) / (batch_output - batch_output**2)
        r_hat_list.append(r_hat)

    return torch.cat(r_hat_list).cpu().numpy().flatten()


def get_r(batch_list):
    x_batch_list = []
    w_batch_list = []
    for sample in batch_list:
        x_batch_list.append(sample[0])
        w_batch_list.append(sample[2])
    x_batch = torch.stack(x_batch_list)
    r_batch = np.sqrt(x_batch[:,0]**2 + x_batch[:,1]**2).reshape(-1, 1)
    w_batch = torch.stack(w_batch_list)
    return r_batch, w_batch


@torch.no_grad()
def get_plot_data(loader):
    temp_x = []
    temp_w = []
    for i, batch in enumerate(loader):
        temp_x.append(batch[0])
        temp_w.append(batch[1])
    return torch.cat(temp_x).numpy().flatten(), torch.cat(temp_w).numpy().flatten()
    



# In[100]:


if __name__ == "__main__":
    batch_size = int(2**8)

    sample_arr = np.load("sample_array.npy")
    sample_ix = int(sys.argv[1])
    sample_size = int(sample_arr[sample_ix])
    print(sample_size, sample_ix)

    source_file = "/data/mdrnevich/ml4nw/NegSampleStudy/base_distribution_data"
    target_file = "/data/mdrnevich/ml4nw/NegSampleStudy/target_distribution_data"
    
    test_base_dataset = utils.preprocessing.Dataset(source_file + "_test.npy", 0)
    test_target_dataset = utils.preprocessing.Dataset(target_file + "_test.npy", 1)
    
    test_base_dataset.process(normalize_weights=True)
    test_target_dataset.process(normalize_weights=True)
    
    test_generator_data = utils.preprocessing.CombinedDataset(test_base_dataset, test_target_dataset)
    
    test_loader = DataLoader(test_generator_data, batch_size=1024, shuffle=False)

    model_path = osp.join("/data/mdrnevich/ml4nw/NegSampleStudy/models/classifier_batch{}_sample{}.zip".format(batch_size, sample_ix))
    #model_path = osp.join("/data/mdrnevich/ml4nw/NegSampleStudy/models/classifier_batch{}_eta{}.zip".format(batch_size, eta_ix))

    mc_model = utils.models.load_model(model_path, device=DEVICE).to(DEVICE)
    mc_X_scaler, mc_weight_norm = utils.preprocessing.load_scaling(model_path)
        
    #mc_test_scores, mc_test_targets, mc_test_weights = get_scores(
    #    mc_model,
    #    test_loader,
    #    X_scaler=mc_X_scaler,
    #    leave=False
    #)

    test_loss = utils.train.test(
        mc_model,
        test_loader,
        X_scaler=mc_X_scaler,
        weight_norm=mc_weight_norm,
        loss='qdre',
        progress_bar=False,
        leave=False,
        device=DEVICE
    )[1]


    test_nominal_loader = DataLoader(utils.preprocessing.CombinedDataset(test_base_dataset), batch_size=batch_size, shuffle=False)
    test_target_loader = DataLoader(utils.preprocessing.CombinedDataset(test_target_dataset), batch_size=batch_size, shuffle=False)
    
    test_nominal_loader.collate_fn = lambda batch: get_r(batch)
    test_target_loader.collate_fn = lambda batch: get_r(batch)

    test_nominal_r = get_plot_data(test_nominal_loader)
    test_target_r = get_plot_data(test_target_loader)

    mc_nominal_ratios = get_r_hats(
        mc_model,
        test_nominal_loader,
        X_scaler=mc_X_scaler,
        weight_norm=mc_weight_norm,
        classifier='new',
        mix=False,
        leave=False
    )
    edges = np.linspace(0, 8, 150)
    divergence = utils.plotting.Tsallis_KL(test_target_r[0], test_target_r[1], test_nominal_r[0], test_nominal_r[1]*mc_nominal_ratios, edges)
    

    rval = np.array((sample_ix,
                     #metrics.roc_auc_score(mc_test_targets, mc_test_scores, sample_weight=mc_test_weights),
                     test_loss,
                     divergence))
    np.save("/data/mdrnevich/ml4nw/NegSampleStudy/temp/results_{}.npy".format(sample_ix), rval)
