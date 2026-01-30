import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import (
    Sequential as Seq,
    Linear as Lin,
    ReLU,
    BatchNorm1d,
    Sigmoid,
)
import onnx, onnxruntime
import numpy as np

import zipfile
import yaml
import collections
import io
from typing import List
import os
import os.path as osp

from . import preprocessing as carl_preprocessing



class Classifier(torch.nn.Module):
    def __init__(self, inputs, hidden_nodes, outputs, final_activation="sigmoid"):
        super(Classifier, self).__init__()
        self.inputs = inputs
        self.hidden_nodes = hidden_nodes
        self.outputs = outputs
        
        layers = []
        layers.append(Lin(inputs, hidden_nodes[0]))
        layers.append(ReLU())
        for i in range(len(hidden_nodes)-1):
            layers.append(Lin(hidden_nodes[i], hidden_nodes[i+1]))
            layers.append(ReLU())
        layers.append(Lin(hidden_nodes[-1], outputs))
        if final_activation == "sigmoid":
            layers.append(Sigmoid())
        self.model = Seq(*layers)
        

    def forward(self, x):
        return self.model(x)


# For learning to predict the weights
class Regression(torch.nn.Module):
    def __init__(self, inputs, hidden_nodes, outputs):
        super(Regression, self).__init__()
        self.inputs = inputs
        self.hidden_nodes = hidden_nodes
        self.outputs = outputs
        self._output_mean_scale = torch.tensor(0, requires_grad=False)
        self._output_std_scale = torch.tensor(1, requires_grad=False)
        
        layers = []
        layers.append(Lin(inputs, hidden_nodes[0]))
        layers.append(ReLU())
        for i in range(len(hidden_nodes)-1):
            layers.append(Lin(hidden_nodes[i], hidden_nodes[i+1]))
            layers.append(ReLU())
        layers.append(Lin(hidden_nodes[-1], outputs))
        self.model = Seq(*layers)
        

    def forward(self, x, transform=True):
        if transform is True:
            return (self.model(x) * self._output_std_scale) + self._output_mean_scale
        else:
            return self.model(x)


    def set_output_scaling(self, mean, std):
        if type(mean) is torch.Tensor:
            self._output_mean_scale = mean.clone().detach()
        else:
            self._output_mean_scale = torch.tensor(mean, requires_grad=False)
        if type(std) is torch.Tensor:
            self._output_std_scale = std.clone().detach()
        else:
            self._output_std_scale = torch.tensor(std, requires_grad=False)


    def get_output_scaling(self):
        return (self._output_mean_scale, self._output_std_scale)



def sigmoid_inv(x):
    return torch.log(x) - torch.log(1-x)



class CalibratedClassifier(torch.nn.Module):
    def __init__(self, classifier, calibrator):
        super(CalibratedClassifier, self).__init__()
        self.classifier = classifier
        self.calibrator = calibrator        

    def forward(self, x):
        return self.calibrator(sigmoid_inv(self.classifier(x)))




# SMM Model with two optimizable coefficients (four subdensities)
class MixtureClassifier(torch.nn.Module):
    def __init__(self, subclassifier_paths, t0=None, t1=None, fine_tune=False):
        """
        In the constructor we instantiate two parameters and assign them as
        member parameters.
        """
        super().__init__()
        
        self.subclassifier_paths = subclassifier_paths
        self.c0 = torch.nn.Parameter(torch.rand(()))
        self.c1 = torch.nn.Parameter(torch.rand(()))
        self.fine_tune = fine_tune
        # Only relevant for PARE score function
        self.t0 = t0
        self.t1 = t1

        self.set_submodels(subclassifier_paths)#, device=device)

    
    @property
    def coefficients(self):
        return (self.c0.cpu().detach().item(), self.c1.cpu().detach().item())


    def set_submodels(self, subclassifier_paths):
        X_scalers, weight_norms = list(zip(*[carl_preprocessing.load_scaling(path) for path in subclassifier_paths]))
        subclassifiers = [load_model(path) for path in subclassifier_paths]

        self.subclassifiers = torch.nn.ModuleList(subclassifiers)
        for module in self.subclassifiers:
            for param in module.parameters():
                param.requires_grad = self.fine_tune
        
        self._scaling = X_scalers
        self.inputs = subclassifiers[0].inputs
        return None


    def set_pare_parameters(self, t0, t1):
        self.t0 = t0
        self.t1 = t1
        return None


    def initialize(self, init):
        self.c0 = torch.nn.Parameter(torch.tensor(init[0], dtype=torch.float))
        self.c1 = torch.nn.Parameter(torch.tensor(init[1], dtype=torch.float))
        return None
    
        
    def forward(self, x, score_function="bce", total_variation=False):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.

        score_function : 'bce', 'mse', 'pare' -> the loss function that determines the optimaal score function
        t0 : setting for 'pare'
        t1 : settting for 'pare'
        """
        
        total_ratio = self.ratio(x, total_variation=total_variation)
        if score_function.lower() in ['bce', 'mse']:
            s = total_ratio / (1 + total_ratio)
        elif score_function.lower() == 'pare':
            if self.t0 is None or self.t1 is None:
                raise Exception("Need to initialize with values for the hyperparemeters t0, t1 to use this score function")
            s = torch.where(torch.isinf(total_ratio), 1/self.t1, (self.t0 + self.t1 * total_ratio) / (self.t0**2 + self.t1**2 * total_ratio))
        elif score_function.lower() == "qdre":
            s = torch.where(total_ratio == 0, 0.5, (total_ratio + 2 - torch.sqrt(total_ratio**2 + 4)) / (2*total_ratio))           
        else:
            raise Exception("Unsupported score function: {}".format(score_function))
        return s
            

    # TOTAL VARIATION NEEDS TO BE IMPLEMENTED
    def ratio(self, x, total_variation=False):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        if total_variation is True:
            raise NotImplementedError("Total variation needs to be implemented!")

        DEVICE = x.device.type
        xx = [torch.from_numpy(self._scaling[i].transform(x.to('cpu')).astype(np.float32)).to(DEVICE) for i in range(len(self.subclassifiers))]
        
        s = [self.subclassifiers[i](xx[i]) for i in range(len(self.subclassifiers))]
        r = [(1-s[i]) / s[i] for i in range(len(self.subclassifiers))]

        # If during optimization one of the coefficients approaches 0 or 1 then things blow up! Need to handle this case separately...        
        part1 = (self.c0 / self.c1) * r[0]
        part2 = ((1 - self.c0) / self.c1) * r[2]
        part3 = (self.c0 / (1 - self.c1)) * r[1]
        part4 = ((1 - self.c0) / (1 - self.c1)) * r[3]
        
        denom1 = torch.where(torch.isinf(part1) & torch.isinf(part2), torch.inf, part1+part2)
        denom2 = torch.where(torch.isinf(part3) & torch.isinf(part4), torch.inf, part3+part4)
        total_ratio = 1/denom1 + 1/denom2
        
        #denom1, denom2 = (part1+part2, part3+part4)
        #total_ratio = torch.where(torch.isnan(denom1), 0, 1/denom1)
        #total_ratio += torch.where(torch.isnan(denom2), 0, 1/denom2)
        #total_ratio = 1/(part1+part2) + 1/(part3+part4)
        return total_ratio # q(Y=1)/q(Y=0)




# SMM Model with only one optimizable coefficient (two subdensities)
class SingleMixtureClassifier(MixtureClassifier):
    def __init__(self, subclassifier_paths, which_mixture=1, t0=None, t1=None, fine_tune=False):
        super().__init__(subclassifier_paths, t0=t0, t1=t1, fine_tune=fine_tune)
        self._which_mixture = which_mixture        
        

    def ratio(self, x, total_variation=False):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.

        which_mixture : int, 0 or 1 -> which class is decomposed as a mixture
        """

        DEVICE = x.device.type
        xx = [torch.from_numpy(self._scaling[i].transform(x.to('cpu')).astype(np.float32)).to(DEVICE) for i in range(len(self.subclassifiers))]
        
        s = [self.subclassifiers[i](xx[i]) for i in range(len(self.subclassifiers))]

        # The source is decomposed as a mixture
        if self._which_mixture == 0:
            r = [(1-s[i]) / s[i] for i in range(len(self.subclassifiers))]
            if total_variation is False:
                return 1 / (self.c0 * r[0] + (1 - self.c0) * r[1])
            elif total_variation is True:
                return (2*self.c0 - 1) / (self.c0 * r[0] + (self.c0 - 1) * r[1])
            else:
                raise Exception("Not a valid choice for 'total_variation': {}".format(total_variation))
        # The target is decomposed as a mixture
        elif self._which_mixture == 1:
            r = [s[i] / (1-s[i]) for i in range(len(self.subclassifiers))]
            if total_variation is False:
                return self.c1 * r[0] + (1 - self.c1) * r[1]
            elif total_variation is True:
                return (self.c1 * r[0] + (self.c1 - 1) * r[1]) / (2*self.c1 - 1)
            else:
                raise Exception("Not a valid choice for 'total_variation': {}".format(total_variation))
        else:
            raise Exception("Not a valid choice for 'which_mixture': {}".format(self._which_mixture))



def load_metadata(path_to_zip):    
    with zipfile.ZipFile(path_to_zip, 'r') as zf:
        name = osp.split(osp.splitext(path_to_zip)[0])[-1]
        try:
            metadata = yaml.load(zf.read("{}_metadata.yaml".format(name)), Loader=yaml.CLoader)
        except KeyError:
            metadata = yaml.load(zf.read("model_metadata.yaml"), Loader=yaml.CLoader)

        return metadata


def load_model_state(path_to_zip, model, device='cpu'):    
    with zipfile.ZipFile(path_to_zip, 'r') as zf:
        name = osp.split(osp.splitext(path_to_zip)[0])[-1]
        try:
            model.load_state_dict(torch.load(io.BytesIO(zf.read("{}.pth".format(name))), map_location=device))
        except KeyError:
            model.load_state_dict(torch.load(io.BytesIO(zf.read("model.pth")), map_location=device))
    return model



def load_model(path_to_zip, device='cpu'):
    metadata = load_metadata(path_to_zip)
    model_metadata = metadata["model"]

    classname = model_metadata["name"]
    if classname == "Classifier":
        model = Classifier(model_metadata["inputs"], model_metadata["hidden_nodes"], model_metadata["outputs"])
    elif classname == "Regression":
        model = Regression(model_metadata["inputs"], model_metadata["hidden_nodes"], model_metadata["outputs"])
        model.set_output_scaling(metadata["scaling"]["output_mean"], metadata["scaling"]["output_std"])
    elif classname == "MixtureClassifier":
        model = MixtureClassifier(model_metadata["subclassifier_paths"], t0=model_metadata["t0"], t1=model_metadata["t1"], fine_tune=model_metadata["fine_tune"])
    elif classname == "SingleMixtureClassifier":
        model = SingleMixtureClassifier(model_metadata["subclassifier_paths"], which_mixture=model_metadata["which_mixture"],
                                        t0=model_metadata["t0"], t1=model_metadata["t1"], fine_tune=model_metadata["fine_tune"])
    else:
        raise Exception("Not implemented for class:", model.__class__.__name__)

    model = load_model_state(path_to_zip, model, device=device)
    return model



def load_onnx_model(path_to_zip, device='cpu'):
    with zipfile.ZipFile(path_to_zip, 'r') as zf:
        onnx_model = onnx.load(io.BytesIO(zf.read("model.onnx")))
    return onnx_model


def load_onnx_session(path_to_zip, device='cpu'):
    with zipfile.ZipFile(path_to_zip, 'r') as zf:
        onnx_model = onnxruntime.InferenceSession(zf.read("model.onnx"),
                                                  providers=['TensorrtExecutionProvider',
                                                             ('CUDAExecutionProvider', {"cudnn_conv_algo_search": "DEFAULT"}),
                                                             'CPUExecutionProvider'])
    return onnx_model