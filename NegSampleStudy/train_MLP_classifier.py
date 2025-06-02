#!/usr/bin/env python
# coding: utf-8

# In[2]:


import sys
sys.path.append('../')
import os.path as osp
import os

import utils
import utils.Camel.equations as equations

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(DEVICE)


# ## Initiate the datasets

# In[24]:


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

for dataset in ["train", "val", "test"]:
    source_data = np.concatenate([np.load(source_x_file + "_" + dataset + ".npy")[:,:-1],
                                  np.load(source_weight_file + "_" + dataset + ".npy").reshape(-1,1)], axis=1)
    #source_data = source_data[source_data[:,-1] !=0]
    np.save(source_file + "_" + dataset + ".npy", source_data)
    
    target_data = np.concatenate([np.load(target_x_file + "_" + dataset + ".npy")[:,:-1],
                                  np.load(target_weight_file + "_" + dataset + ".npy").reshape(-1,1)], axis=1)
    #target_data = target_data[target_data[:,-1] !=0]
    np.save(target_file + "_" + dataset + ".npy", target_data)
    #assert source_data.shape[1] == target_data.shape[1]
"""

source_file = "/data/mdrnevich/ml4nw/NegSampleStudy/data/base_distribution_sample_size{:d}".format(sample_size)
target_file = "/data/mdrnevich/ml4nw/NegSampleStudy/data/target_distribution_sample_size{:d}".format(sample_size)


train_base_dataset = utils.preprocessing.Dataset(source_file + "_train.npy", 0)
valid_base_dataset = utils.preprocessing.Dataset(source_file + "_val.npy", 0)

train_target_dataset = utils.preprocessing.Dataset(target_file + "_train.npy", 1)
valid_target_dataset = utils.preprocessing.Dataset(target_file + "_val.npy", 1)

LOSS = "qdre"

training_settings.update({
    "source_file": source_file,
    "target_file": target_file,
    "source_mixture_coef": source_mixture_coef,
    "source_scales": source_scales,
    "target_mixture_coef": target_mixture_coef,
    "target_scales": target_scales,
    "loss": LOSS
})


# ## Load the data

# In[25]:


source_weight_norm = train_base_dataset.process(normalize_weights=True)
#print(source_weight_norm)
source_valid_weight_norm = valid_base_dataset.process(normalize_weights=True)

target_weight_norm = train_target_dataset.process(normalize_weights=True)
valid_target_dataset.process(normalize_weights=True)



train_generator_data = utils.preprocessing.CombinedDataset(train_base_dataset, train_target_dataset)
valid_generator_data = utils.preprocessing.CombinedDataset(valid_base_dataset, valid_target_dataset)



X_scaler, train_weight_norm = utils.preprocessing.get_scaling(train_generator_data)
_, valid_weight_norm = utils.preprocessing.get_scaling(valid_generator_data)
print(train_weight_norm, valid_weight_norm)


# ## Prepare the data for training

random_seed = 0

torch.manual_seed(random_seed)
batch_size = int(2**8)
training_settings.update({
    "batch_size": batch_size,
    "random_seed": random_seed
})

train_loader = DataLoader(train_generator_data, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(valid_generator_data, batch_size=batch_size, shuffle=False)


# ## Construct the model


inputs = 2
hidden_nodes = [64, 64]
outputs = 1

model = utils.models.Classifier(inputs, hidden_nodes, outputs)
print(model)
print("----------")

model = model.to(DEVICE)

learning_rate = 3e-5
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Get the analytical optimal classifier
def qdre_score_function(ratio):
    return (ratio + 2 - np.sqrt(ratio**2 + 4)) / (2*ratio)

r_optimal = equations.optimal_likelihood_ratio(source_mixture_coef, source_scales, target_mixture_coef, target_scales)
s_optimal = lambda x: qdre_score_function(r_optimal(x))


training_settings.update({
    "learning_rate": learning_rate,
    "optimizer": "Adam"
})

model_path = osp.join("/data/mdrnevich/ml4nw/NegSampleStudy/models/classifier_batch{}_sample{}".format(batch_size, sample_ix))


# ## Perform the training

# In[ ]:


n_epochs = 1500
stale_epochs = 0
best_valid_loss = 99999
patience = 20
max_num_batches = int(int(1e5) / batch_size)
t = tqdm(range(0, n_epochs))
print("Starting training")

training_losses = [utils.train.test(
        model,
        train_loader,
        X_scaler=X_scaler,
        loss=LOSS,
        weight_norm=train_weight_norm,
        max_num_batches=max_num_batches,
        device=DEVICE,
        progress_bar=False,
        leave=False
    )[0],]
validation_losses = [utils.train.test(
        model,
        valid_loader,
        X_scaler=X_scaler,
        loss=LOSS,
        weight_norm=valid_weight_norm,
        device=DEVICE,
        progress_bar=False,
        leave=False
    )[0],]


optimal_train_loss = utils.train.get_optimal_loss(
        s_optimal,
        train_loader,
        loss=LOSS,
        weight_norm=train_weight_norm,
        device=DEVICE,
        progress_bar=False
)
optimal_valid_loss = utils.train.get_optimal_loss(
        s_optimal,
        valid_loader,
        loss=LOSS,
        weight_norm=valid_weight_norm,
        device=DEVICE,
        progress_bar=False
)

    
training_settings.update({
    "optimal_train_loss": optimal_train_loss,
    "optimal_valid_loss": optimal_valid_loss
})


for epoch in t:
    loss = utils.train.train(
        model,
        optimizer,
        train_loader,
        X_scaler=X_scaler,
        loss=LOSS,
        weight_norm=train_weight_norm,
        max_num_batches=max_num_batches,
        device=DEVICE,
        leave=bool(epoch == n_epochs - 1),
        progress_bar=False,
    )
    #loss -= optimal_train_loss
    training_losses.append(loss[0])

    valid_loss = utils.train.test(
        model,
        valid_loader,
        X_scaler=X_scaler,
        loss=LOSS,
        weight_norm=valid_weight_norm,
        device=DEVICE,
        leave=bool(epoch == n_epochs - 1),
        progress_bar=False,
    )
    #valid_loss -= optimal_valid_loss
    validation_losses.append(valid_loss[0])
    print("Epoch: {:02d}, Training Loss:   {:.4f}".format(epoch, loss[1]))
    print("           Validation Loss: {:.4f}".format(valid_loss[1]))

    if valid_loss[1] < best_valid_loss:
        best_valid_loss = valid_loss[1]
        training_settings.update({
            "n_epochs": epoch+1,
            "training_losses": training_losses,
            "validation_losses": validation_losses
        })
        model_metadata = utils.train.get_model_metadata(training_settings, model, X_scaler, train_weight_norm)
        utils.train.save_model_data(model, model_metadata, name=model_path, save_onnx=False, device=DEVICE)
        print("New best model saved to: {}.zip".format(model_path))
        #torch.save(model.state_dict(), modpath)
        stale_epochs = 0
    else:
        print("Stale epoch")
        stale_epochs += 1
    if stale_epochs >= patience:
        print("Early stopping after %i stale epochs" % patience)
        break

#for dataset in ["train", "val", "test"]:
#    os.remove(source_file + "_" + dataset + ".npy")
#    os.remove(target_file + "_" + dataset + ".npy")

print("Finished")
