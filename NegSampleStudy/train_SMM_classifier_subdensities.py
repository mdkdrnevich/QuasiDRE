#!/usr/bin/env python
# coding: utf-8

# In[10]:


import sys
sys.path.append('../')
import os.path as osp
import os

import torch
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

import utils
import utils.Camel.equations as equations

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#DEVICE = 'cpu'
print(DEVICE)


# ## Look at the data

# In[3]:


training_settings = {}

"""
These base datasets have settings:
mixture_coef = (4, -1)  # coefficients of the pdf terms
scales = (2.5, 2.5)  # input scaling (x -> x/scale)

These target datasets have settings:
mixture_coef = (2, -1)
scales = (2, 1.42)
with weight_scales=(1,1) and weight spreads=(0,0)

In this case both distributions are sampled directly such that their weights are all +1.
"""

source_mixture_coef = (4, -1)
source_scales = (2.5, 2.3)
target_mixture_coef = (2, -1)
target_scales = (2, 1.2)

sample_arr = np.load("sample_array.npy")
sample_ix = int(sys.argv[1])
sample_size = int(sample_arr[sample_ix])
print(sample_size, sample_ix)

"""
source_file = "/data/mdrnevich/ml4nw/NegSampleStudy/data/base_distribution_data_sample_size{}".format(sample_ix)
target_file = "/data/mdrnevich/ml4nw/NegSampleStudy/data/target_distribution_data_sample_size{}".format(sample_ix)

source_x_file = "/data/mdrnevich/ml4nw/NegSampleStudy/base_distribution_data"
target_x_file = "/data/mdrnevich/ml4nw/NegSampleStudy/target_distribution_data"
source_weight_file = "/data/mdrnevich/ml4nw/NegSampleStudy/weights/base_distribution_weights_sample_size{:d}".format(sample_size)
target_weight_file = "/data/mdrnevich/ml4nw/NegSampleStudy/weights/target_distribution_weights_sample_size{:d}".format(sample_size)
"""

source_file = "/data/mdrnevich/ml4nw/NegSampleStudy/data/base_distribution_sample_size{:d}".format(sample_size)
target_file = "/data/mdrnevich/ml4nw/NegSampleStudy/data/target_distribution_sample_size{:d}".format(sample_size)


SAVE = True

if SAVE is True:
    for dataset in ["train", "val", "test"]:
        #arr = np.concatenate([np.load(source_x_file + "_" + dataset + ".npy")[:,:-1],
        #                      np.load(source_weight_file + "_" + dataset + ".npy").reshape(-1,1)], axis=1)
        arr = np.load(source_file + "_" + dataset + ".npy")
        np.save(source_file + "_positives_" + dataset + ".npy",
                arr[arr[:,-1] >= 0])
        np.save(source_file + "_negatives_" + dataset + ".npy",
                arr[arr[:,-1] < 0])
        
        #arr = np.concatenate([np.load(target_x_file + "_" + dataset + ".npy")[:,:-1],
        #                      np.load(target_weight_file + "_" + dataset + ".npy").reshape(-1,1)], axis=1)
        arr = np.load(target_file + "_" + dataset + ".npy")
        np.save(target_file + "_positives_" + dataset + ".npy",
                arr[arr[:,-1] >= 0])
        np.save(target_file + "_negatives_" + dataset + ".npy",
                arr[arr[:,-1] < 0])


# In[5]:


source_positive_file = source_file + "_positives"
source_negative_file = source_file + "_negatives"
target_positive_file = target_file + "_positives"
target_negative_file = target_file + "_negatives"

files = [source_positive_file, source_negative_file, target_positive_file, target_negative_file]

combos = [np.array((0,2)), #++
          np.array((0,3)), #+-
          np.array((1,2)), #-+
          np.array((1,3))] #--

train_datasizes = []
val_datasizes = []
test_datasizes = []
for f in files:
    train_datasizes.append(np.load(f + "_train.npy").shape[0])
    val_datasizes.append(np.load(f + "_val.npy").shape[0])
    test_datasizes.append(np.load(f + "_test.npy").shape[0])

train_datasizes = np.array(train_datasizes)
val_datasizes = np.array(val_datasizes)
test_datasizes = np.array(test_datasizes)
print(train_datasizes)


# ## Initiate the datasets

# In[6]:


training_settings = [{}, {}, {}, {}]

"""
These base datasets have settings:
mixture_coef = (4, -1)  # coefficients of the pdf terms
scales = (2.5, 2.5)  # input scaling (x -> x/scale)

These target datasets have settings:
mixture_coef = (2, -1)
scales = (2, 1.42)
with weight_scales=(1,1) and weight spreads=(0.4,0.75)

the negative target dataset has scale (2, 1.2) with the same coefficients and weight settings

In this case both distributions are sampled directly such that their weights are all +1.
"""

all_mixture_coef = np.array([*source_mixture_coef, *target_mixture_coef])
all_scales = np.array([*source_scales, *target_scales])

train_generator_datas = []
valid_generator_datas = []
test_generator_datas = []

# If we want to use different training sizes for each combo
for i in range(4):
    min_datasizes = (train_datasizes[combos[i]].min(), val_datasizes[combos[i]].min(), test_datasizes[combos[i]].min())
    print(min_datasizes)
    training_settings[i].update({
        "source_file": files[combos[i][0]],
        "target_file": files[combos[i][1]],
        "n_train": int(min_datasizes[0]),
        "n_val": int(min_datasizes[1]),
        "n_test": int(min_datasizes[2]),
        "source_mixture_coef": (1,0),
        "source_scales": (all_scales[combos[i][0]], all_scales[combos[i][0]]),
        "target_mixture_coef": (1,0),
        "target_scales": (all_scales[combos[i][1]], all_scales[combos[i][1]])
    })


    # We flip the labels here so we learn the ratio of Y=0/Y=1 when we use the regular s/(1-s) trick, which is what we want for the subdensities
    train_base_dataset = utils.preprocessing.Dataset(files[combos[i][0]] + "_train.npy", 0,
                                                     stop_event=training_settings[i]["n_train"])
    valid_base_dataset = utils.preprocessing.Dataset(files[combos[i][0]] + "_val.npy", 0,
                                                     stop_event=training_settings[i]["n_val"])

    train_target_dataset = utils.preprocessing.Dataset(files[combos[i][1]] + "_train.npy", 1,
                                                     stop_event=training_settings[i]["n_train"])
    valid_target_dataset = utils.preprocessing.Dataset(files[combos[i][1]] + "_val.npy", 1,
                                                       stop_event=training_settings[i]["n_val"])


    source_weight_norm = train_base_dataset.process(normalize_weights=True)
    source_valid_weight_norm = valid_base_dataset.process(normalize_weights=True)
    
    target_weight_norm = train_target_dataset.process(normalize_weights=True)
    target_valid_weight_norm = valid_target_dataset.process(normalize_weights=True)

    train_generator_datas.append(utils.preprocessing.CombinedDataset(train_base_dataset, train_target_dataset))
    valid_generator_datas.append(utils.preprocessing.CombinedDataset(valid_base_dataset, valid_target_dataset))


# ## Do some data preprocessing for standardized inputs and weights

# In[7]:


X_scalers, train_weight_norms = list(zip(*[utils.preprocessing.get_scaling(train_generator_data) for train_generator_data in train_generator_datas]))
_, valid_weight_norms = list(zip(*[utils.preprocessing.get_scaling(valid_generator_data) for valid_generator_data in valid_generator_datas]))
print(train_weight_norms, valid_weight_norms)


# ## Prepare the data for training

# In[8]:


random_seed = 0

torch.manual_seed(random_seed)
batch_size = int(2**8)
[ts.update({
    "batch_size": batch_size,
    "random_seed": random_seed
}) for ts in training_settings]

train_loaders = [DataLoader(train_generator_data, batch_size=batch_size, shuffle=True) for train_generator_data in train_generator_datas]
valid_loaders = [DataLoader(valid_generator_data, batch_size=batch_size, shuffle=False) for valid_generator_data in valid_generator_datas]


# ## Construct the model

# In[9]:


inputs = 2
hidden_nodes = [32, 32]
outputs = 1

models = [utils.models.Classifier(inputs, hidden_nodes, outputs).to(DEVICE) for _ in range(4)]
print(models)
print("----------")

#model = model.to(DEVICE)

learning_rate = 1e-4
optimizers = [torch.optim.Adam(model.parameters(), lr=learning_rate) for model in models]

# Get the analytical optimal classifier
s_optimals = [equations.optimal_binary_classifier(training_settings[i]["source_mixture_coef"],
                                                  training_settings[i]["source_scales"],
                                                  training_settings[i]["target_mixture_coef"],
                                                  training_settings[i]["target_scales"]) for i in range(4)]

[ts.update({
    "learning_rate": learning_rate,
    "optimizer": "Adam"
}) for ts in training_settings]


# ## Perform the training

# In[ ]:


for i in range(4):
    modpath = osp.join("/data/mdrnevich/ml4nw/NegSampleStudy/models/classifier_subdensity_{}_batch{}_sample{}".format(i+1, batch_size, sample_ix))
    model = models[i]
    optimizer = optimizers[i]
    X_scaler = X_scalers[i]
    train_weight_norm = train_weight_norms[i]
    valid_weight_norm = valid_weight_norms[i]
    train_loader = train_loaders[i]
    valid_loader = valid_loaders[i]
    s_optimal = s_optimals[i]
    
    n_epochs = 1500
    stale_epochs = 0
    best_valid_loss = 99999
    patience = 20
    max_num_batches = int(int(1e5) / batch_size)
    t = tqdm(range(0, n_epochs))
    
    training_losses = [utils.train.test(
            model,
            train_loader,
            X_scaler,
            weight_norm=train_weight_norm,
            max_num_batches=max_num_batches,
            device=DEVICE,
            progress_bar=False,
            leave=False
        )[0],]
    validation_losses = [utils.train.test(
            model,
            valid_loader,
            X_scaler,
            weight_norm=valid_weight_norm,
            device=DEVICE,
            progress_bar=False,
            leave=False
        )[0],]
    
    optimal_train_loss = utils.train.get_optimal_loss(
            s_optimal,
            train_loader,
            weight_norm=train_weight_norm,
            device=DEVICE,
            progress_bar=False
    )
    optimal_valid_loss = utils.train.get_optimal_loss(
            s_optimal,
            valid_loader,
            weight_norm=valid_weight_norm,
            device=DEVICE,
            progress_bar=False
    )
    training_settings[i].update({
        "optimal_train_loss": optimal_train_loss,
        "optimal_valid_loss": optimal_valid_loss
    })
    
    
    for epoch in t:
        loss = utils.train.train(
            model,
            optimizer,
            train_loader,
            X_scaler,
            weight_norm=train_weight_norm,
            max_num_batches=max_num_batches,
            device=DEVICE,
            progress_bar=False,
            leave=bool(epoch == n_epochs - 1),
        )
        #loss -= optimal_train_loss
        training_losses.append(loss[0])
    
        valid_loss = utils.train.test(
            model,
            valid_loader,
            X_scaler,
            weight_norm=valid_weight_norm,
            device=DEVICE,
            progress_bar=False,
            leave=bool(epoch == n_epochs - 1),
        )
        #valid_loss -= optimal_valid_loss
        validation_losses.append(valid_loss[0])
        print("Epoch: {:02d}, Training Loss:   {:.4f}".format(epoch, loss[1]))
        print("           Validation Loss: {:.4f}".format(valid_loss[1]))
    
        if valid_loss[1] < best_valid_loss:
            best_valid_loss = valid_loss[1]
            training_settings[i].update({
                "n_epochs": epoch+1,
                "training_losses": training_losses,
                "validation_losses": validation_losses
            })
            model_metadata = utils.train.get_model_metadata(training_settings[i], model, X_scaler, train_weight_norm)
            utils.train.save_model_data(model, model_metadata, name=modpath, save_onnx=False, device=DEVICE)
            print("New best model saved to: {}.zip".format(modpath))
            #torch.save(model.state_dict(), modpath)
            stale_epochs = 0
        else:
            print("Stale epoch")
            stale_epochs += 1
        if stale_epochs >= patience:
            print("Early stopping after %i stale epochs" % patience)
            break



for dataset in ["train", "val", "test"]:
    os.remove(source_file + "_positives_" + dataset + ".npy")
    os.remove(source_file + "_negatives_" + dataset + ".npy")
    os.remove(target_file + "_positives_" + dataset + ".npy")
    os.remove(target_file + "_negatives_" + dataset + ".npy")

