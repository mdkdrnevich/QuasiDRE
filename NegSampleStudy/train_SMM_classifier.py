#!/usr/bin/env python
# coding: utf-8

# In[25]:


import sys
sys.path.append('../')
import os.path as osp
import os

import torch
from torch.utils.data import DataLoader
import numpy as np
import copy
from tqdm import tqdm
import matplotlib.pyplot as plt

import utils
import utils.Camel.equations as equations

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#DEVICE = 'cpu'
print(DEVICE)


# # Now create a combined model and optimize the coefficients


sample_arr = np.load("sample_array.npy")
sample_ix = int(sys.argv[1])
sample_size = int(sample_arr[sample_ix])
print(sample_size, sample_ix)


SAVE_SMM = False
FINE_TUNE = True

t0 = 25619
t1 = 58
print(t0, t1, -(t0/t1)**2, -t0/t1)


batch_size = int(2**8)
model_paths = [osp.join("/data/mdrnevich/ml4nw/NegSampleStudy/models/classifier_subdensity_{}_batch{}_sample{}.zip".format(i+1, batch_size, sample_ix)) for i in range(4)]
mix_model = utils.models.MixtureClassifier(model_paths, t0=t0, t1=t1, fine_tune=FINE_TUNE).to(DEVICE)


# In[14]:

source_file = "/data/mdrnevich/ml4nw/NegSampleStudy/data/base_distribution_sample_size{:d}".format(sample_size)
target_file = "/data/mdrnevich/ml4nw/NegSampleStudy/data/target_distribution_sample_size{:d}".format(sample_size)

files = [source_file, target_file]

#sum_weights = []
#for f in files:
#    weights = np.load(f + "_train.npy")[:,-1]
#    sum_weights.append(weights[weights >= 0].sum())
#    sum_weights.append(weights[weights < 0].sum())

#sum_weights = np.array(sum_weights)
#sum_weights


# In[15]:


#coefficient_init = [sum_weights[0] / sum_weights[:2].sum(), sum_weights[2] / sum_weights[2:].sum()]
#coefficient_init


coefficient_init = []
for f in files:
    weights = np.load(f + "_train.npy")[:,-1]
    coefficient_init.append(weights[weights > 0].sum()  / weights.sum())
print("Coefficient init")
print(coefficient_init)



# In[13]:


mix_model.initialize(coefficient_init)
list(mix_model.parameters())


# In[16]:


training_settings_mix = {}

source_mixture_coef = (4, -1)
source_scales = (2.5, 2.3)
target_mixture_coef = (2, -1)
target_scales = (2, 1.42)


MIN_TRAIN_SIZE = int(min([np.load(source_file + "_train.npy").shape[0], np.load(target_file + "_train.npy").shape[0]]))
MIN_VALID_SIZE = int(min([np.load(source_file + "_val.npy").shape[0], np.load(target_file + "_val.npy").shape[0]]))

train_base_dataset = utils.preprocessing.Dataset(source_file + "_train.npy", 0, stop_event=MIN_TRAIN_SIZE)
valid_base_dataset = utils.preprocessing.Dataset(source_file + "_val.npy", 0, stop_event=MIN_VALID_SIZE)

train_target_dataset = utils.preprocessing.Dataset(target_file + "_train.npy", 1, stop_event=MIN_TRAIN_SIZE)
valid_target_dataset = utils.preprocessing.Dataset(target_file + "_val.npy", 1, stop_event=MIN_VALID_SIZE)

training_settings_mix.update({
    "source_file": source_file,
    "target_file": target_file,
    "source_mixture_coef": source_mixture_coef,
    "source_scales": source_scales,
    "target_mixture_coef": target_mixture_coef,
    "target_scales": target_scales,
    "t0": t0,
    "t1": t1
})


# In[17]:


source_weight_norm = train_base_dataset.process(normalize_weights=True)
valid_base_dataset.process(normalize_weights=True)

target_weight_norm = train_target_dataset.process(normalize_weights=True)
valid_target_dataset.process(normalize_weights=True)


# In[18]:


train_generator_data = utils.preprocessing.CombinedDataset(train_base_dataset, train_target_dataset)
valid_generator_data = utils.preprocessing.CombinedDataset(valid_base_dataset, valid_target_dataset)


# In[19]:


X_scaler, train_weight_norm = utils.preprocessing.get_scaling(train_generator_data)
_, valid_weight_norm = utils.preprocessing.get_scaling(valid_generator_data)
print(train_weight_norm, valid_weight_norm)


if SAVE_SMM is True:
    SMM_model = utils.models.MixtureClassifier(model_paths, t0=t0, t1=t1, fine_tune=False).to(DEVICE)
    SMM_model.initialize(coefficient_init)
    model_path = osp.join("/data/mdrnevich/ml4nw/NegSampleStudy/models/classifier_SMM_batch{}_sample{}".format(batch_size, sample_ix))
    model_metadata = utils.train.get_model_metadata(training_settings_mix, SMM_model, X_scaler, train_weight_norm)
    utils.train.save_model_data(SMM_model, model_metadata, name=model_path, save_onnx=False, device=DEVICE)


# In[20]:


random_seed = 0

torch.manual_seed(random_seed)

train_loader = DataLoader(train_generator_data, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(valid_generator_data, batch_size=batch_size, shuffle=False)


# In[22]:


learning_rate = 1e-4
optimizer = torch.optim.Adam(mix_model.parameters(), lr=learning_rate)

# Get the analytical optimal classifier
#s_optimal = equations.optimal_binary_classifier(source_mixture_coef, source_scales, target_mixture_coef, target_scales)

# Get the analytical optimal classifier
def qdre_score_function(ratio):
    return (ratio + 2 - np.sqrt(ratio**2 + 4)) / (2*ratio)

r_optimal = equations.optimal_likelihood_ratio(source_mixture_coef, source_scales, target_mixture_coef, target_scales)
s_optimal = lambda x: qdre_score_function(r_optimal(x))


if FINE_TUNE is False:
    model_path = osp.join("/data/mdrnevich/ml4nw/NegSampleStudy/models/classifier_SMMc_batch{}_sample{}".format(batch_size, sample_ix))
else:
    model_path = osp.join("/data/mdrnevich/ml4nw/NegSampleStudy/models/classifier_SMMr_batch{}_sample{}".format(batch_size, sample_ix))



# In[23]:


n_epochs = 1500
stale_epochs = 0
best_valid_loss = 99999
patience = 10
max_num_batches = int(int(1e5) / batch_size)
t = tqdm(range(0, n_epochs))

training_losses = [utils.train.test(
        mix_model,
        train_loader,
        weight_norm=train_weight_norm,
        loss='qdre',
        SMM=True,
        t0=t0,
        t1=t1,
        device=DEVICE,
        max_num_batches=max_num_batches,
        progress_bar=False,
        leave=False
    )[0],]
validation_losses = [utils.train.test(
        mix_model,
        valid_loader,
        weight_norm=valid_weight_norm,
        loss='qdre',
        SMM=True,
        t0=t0,
        t1=t1,
        device=DEVICE,
        progress_bar=False,
        leave=False
    )[0],]

optimal_train_loss = utils.train.get_optimal_loss(
        s_optimal,
        train_loader,
        weight_norm=train_weight_norm,
        loss='qdre',
        t0=t0,
        t1=t1,
        device=DEVICE,
        progress_bar=False
)
optimal_valid_loss = utils.train.get_optimal_loss(
        s_optimal,
        valid_loader,
        weight_norm=valid_weight_norm,
        loss='qdre',
        t0=t0,
        t1=t1,
        device=DEVICE,
        progress_bar=False
)

training_settings_mix.update({
    "optimal_train_loss": optimal_train_loss,
    "optimal_valid_loss": optimal_valid_loss
})

for epoch in t:
    loss = utils.train.train(
        mix_model,
        optimizer,
        train_loader,
        weight_norm=train_weight_norm,
        max_num_batches=max_num_batches,
        loss='qdre',
        SMM=True,
        t0=t0,
        t1=t1,
        device=DEVICE,
        progress_bar=False,
        leave=bool(epoch == n_epochs - 1),
    )
    #loss -= optimal_train_loss
    training_losses.append(loss[0])

    valid_loss = utils.train.test(
        mix_model,
        valid_loader,
        weight_norm=valid_weight_norm,
        loss='qdre',
        SMM=True,
        t0=t0,
        t1=t1,
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
        print("New best model with parameters:", mix_model.coefficients)
        training_settings_mix.update({
            "n_epochs": epoch+1,
            "training_losses": training_losses,
            "validation_losses": validation_losses
        })
        model_metadata = utils.train.get_model_metadata(training_settings_mix, mix_model, X_scaler, train_weight_norm)
        utils.train.save_model_data(mix_model, model_metadata, name=model_path, save_onnx=False, device=DEVICE)
        print("New best model saved to: {}.zip".format(model_path))
        stale_epochs = 0
    else:
        print("Stale epoch")
        stale_epochs += 1
    if stale_epochs >= patience:
        print("Early stopping after %i stale epochs" % patience)
        break


# In[26]:

print("Finished")




