import torch
import numpy as np
import os.path as osp
import tempfile
import shutil
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from tqdm import tqdm
import vector
import utils.plotting
import types


@torch.no_grad()
def get_scores(model, loader, X_scaler=None, weight_norm=1, mix=False, leave=False, device='cpu'):
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
    t = tqdm(enumerate(loader), total=len(loader), leave=leave)
    for i, batch in t:
        target_list.append(batch[1])
        weight_list.append(batch[2])
        if type(model) is not types.FunctionType:
            x = batch[0].to(device)
        else:
            x = batch[0].to('cpu')
        batch_score = model(x)
        score_list.append(batch_score)
        t.refresh()  # to show immediately the update

    return torch.cat(score_list).cpu().numpy().flatten(), torch.cat(target_list).cpu().numpy().flatten(), torch.cat(weight_list).cpu().numpy().flatten()


@torch.no_grad()
def get_r_hats(model, loader, X_scaler=None, weight_norm=1, mix=False, leave=False, loss="bce", t0=None, t1=None, device='cpu'):
    if type(model) is not types.FunctionType:
        model.eval()
        if mix is True:
            loader.collate_fn = lambda batch: utils.preprocessing.prep_inputs_for_training_mix(batch, weight_norm=weight_norm)
        else:
            loader.collate_fn = lambda batch: utils.preprocessing.prep_inputs_for_training(batch, X_scaler, weight_norm=weight_norm)
    else:
        loader.collate_fn = lambda batch: utils.preprocessing.prep_inputs_for_density(batch, weight_norm=weight_norm)

    r_hat_list = []
    t = tqdm(enumerate(loader), total=len(loader), leave=leave)
    for i, batch in t:
        if type(model) is not types.FunctionType:
            x = batch[0].to(device)
        else:
            x = batch[0].to('cpu')
        
        if mix is True:
            r_hat = model.ratio(x)
        else:
            batch_output = model(x)
            if loss in ["bce", "mse"]:
                r_hat = batch_output / (1 - batch_output)
            elif loss == "pare":
                r_hat = - t0*(1- t0*batch_output)/(t1*(1 - t1*batch_output))
            elif loss == 'revert':
                r_hat = (1-2*batch_output) / (batch_output - batch_output**2)
        r_hat_list.append(r_hat)
        t.refresh()  # to show immediately the update

    return torch.cat(r_hat_list).cpu().numpy().flatten()


def get_x_i(batch_list, ix):
    x_batch_list = []
    w_batch_list = []
    for sample in batch_list:
        x_batch_list.append(sample[0])
        w_batch_list.append(sample[2])
    x_batch = torch.stack(x_batch_list)[:,ix]
    w_batch = torch.stack(w_batch_list)
    return x_batch, w_batch


def get_HH_4vec(batch_list, keys, nMuons=4):
    MUON_PT_IX = keys.index('Muon_Pt0')
    MUON_ETA_IX = keys.index('Muon_Eta0')
    MUON_PHI_IX = keys.index('Muon_Phi0')
    #MUON_M_IX = keys.index('Muon_Mass0')
    
    x_batch_list = []
    w_batch_list = []
    for sample in batch_list:
        x_batch_list.append(sample[0])
        w_batch_list.append(sample[2])
    x_batch = torch.stack(x_batch_list).numpy()
    w_batch = torch.stack(w_batch_list)

    ZEROS = np.zeros_like(x_batch[:, 0])
    HH_4vec = vector.array({"pt":ZEROS,"eta":ZEROS,"phi":ZEROS,"M":ZEROS})
    for i in range(nMuons):
        HH_4vec += vector.array({
            "pt": x_batch[:, MUON_PT_IX+i],
            "eta": x_batch[:, MUON_ETA_IX+i],
            "phi": x_batch[:, MUON_PHI_IX+i],
            #"M": x_batch[:, MUON_M_IX+i]
            "M": np.ones_like(x_batch[:, MUON_PT_IX+i])*0.10566001 #MUON MASS IS FIXED IN THIS DATA
        })
    return HH_4vec, w_batch

def get_mHH(batch_list, keys, nMuons=4):
    HH_4vec, w_batch = get_HH_4vec(batch_list, keys, nMuons=nMuons)
    return torch.from_numpy(HH_4vec.M), w_batch

def get_pt_HH(batch_list, keys, nMuons=4):
    HH_4vec, w_batch = get_HH_4vec(batch_list, keys, nMuons=nMuons)
    return torch.from_numpy(HH_4vec.pt), w_batch

def get_eta_HH(batch_list, keys, nMuons=4):
    HH_4vec, w_batch = get_HH_4vec(batch_list, keys, nMuons=nMuons)
    return torch.from_numpy(HH_4vec.eta), w_batch

def get_phi_HH(batch_list, keys, nMuons=4):
    HH_4vec, w_batch = get_HH_4vec(batch_list, keys, nMuons=nMuons)
    return torch.from_numpy(HH_4vec.phi), w_batch


@torch.no_grad()
def get_plot_data(loader, leave=False):
    temp_x = []
    temp_w = []
    t = tqdm(enumerate(loader), total=len(loader), leave=leave)
    for i, batch in t:
        temp_x.append(batch[0])
        temp_w.append(batch[1])
        t.refresh()  # to show immediately the update
    return torch.cat(temp_x).numpy().flatten(), torch.cat(temp_w).numpy().flatten()


def plot_closure(ratio_est,
                 nominal_loader,
                 target_loader,
                 model_name,
                 target_name,
                 keys,
                 feature_group='muon_pt',
                 nMuons=4,
                 percentile_cuts=(0.0, 100.0),
                 carl_names=None,
                 kwargs_plot=None,
                 figsize=(16,12),
                 show=True,
                 save=True):
    """
    Plot CARL reweighting closure for a subdensity (wraps the code from the previous cell).
    - model_idx: index of the subdensity (0..NUM_MODELS-1)
    - feature_group: one of 'muon_pt', 'muon_phi', 'muon_eta', 'jet' (all jets on one image),
                     or one of 'jet_pt', 'jet_eta', 'jet_phi', 'jet_mass' (single jet feature)
    - nMuons: number of muons (default 4)
    - percentile_cuts: tuple for plot_distributions
    - carl_names: list of names for CARL curves (defaults to ["<name> CARL"])
    - kwargs_plot: dict of plotting kwargs (font sizes etc.), defaults to the notebook kwargs
    - show: whether to display images inline
    - save: whether to save temporary images (if False will just call plot_distributions)
    Returns list of saved image paths (empty list if save=False).
    """
    if carl_names is None:
        carl_names = [f"{model_name} CARL"]

    if kwargs_plot is None:
        kwargs_plot = {'legend.title_fontsize': 18, 'legend.fontsize': 14,
                        'font.size': 14, 'axes.titlesize':20, 'axes.labelsize':16,'figure.titlesize':20, 'ytick.labelsize':12}

    saved_paths = []

    # mapping for groups to base index + number of plots
    if feature_group == 'muon_pt':
        base_idx = 4 + nMuons * 2
        n_plots = nMuons
        feature_labels = ["1st Muon $p_T$ [GeV]", "2nd Muon $p_T$ [GeV]", "3rd Muon $p_T$ [GeV]", "4th Muon $p_T$ [GeV]"]
    elif feature_group == 'muon_phi':
        base_idx = 4 + nMuons
        n_plots = nMuons
        feature_labels = ["1st Muon $\\phi$", "2nd Muon $\\phi$", "3rd Muon $\\phi$", "4th Muon $\\phi$"]
    elif feature_group == 'muon_eta':
        base_idx = 4
        n_plots = nMuons
        feature_labels = ["1st Muon $\\eta$", "2nd Muon $\\eta$", "3rd Muon $\\eta$", "4th Muon $\\eta$"]
    elif feature_group == 'jet_pt':
        base_idx = 3
        n_plots = 1
        feature_labels = ["Jet $p_T$ [GeV]"]
    elif feature_group == 'jet_eta':
        base_idx = 0
        n_plots = 1
        feature_labels = ["Jet $\\eta$"]
    elif feature_group == 'jet_phi':
        base_idx = 2
        n_plots = 1
        feature_labels = ["Jet $\\phi$"]
    elif feature_group == 'jet_mass':
        base_idx = 1
        n_plots = 1
        feature_labels = ["Jet Mass [GeV]"]
    elif feature_group == 'jet':
        # show all four jet features on one image (order corresponds to indices 0..3)
        base_idx = 0
        n_plots = 4
        feature_labels = ["Jet $\\eta$", "Jet Mass [GeV]", "Jet $\\phi$", "Jet $p_T$ [GeV]"]
    elif feature_group == 'higgs':
        base_idx = 0
        n_plots = 4
        feature_labels = ["HH $p_T$ [GeV]", "HH $\\eta$", "HH $\\phi$", "$m_{HH}$ [GeV]"]
        # override collate functions for mHH, pT, eta, phi
        hh_loaders = [lambda batch: get_pt_HH(batch, keys, nMuons=nMuons),
                      lambda batch: get_eta_HH(batch, keys, nMuons=nMuons),
                      lambda batch: get_phi_HH(batch, keys, nMuons=nMuons),
                      lambda batch: get_mHH(batch, keys, nMuons=nMuons)]
        
    else:
        raise ValueError(f"Unknown feature_group: {feature_group}")

    # temporary dir for saved images
    if save:
        tmp_image_dir = tempfile.mkdtemp()
        _paths = []

    for j in range(n_plots):
        idx = base_idx + j if n_plots > 1 else base_idx

        if feature_group == 'higgs':
            nominal_loader.collate_fn = hh_loaders[j]
            target_loader.collate_fn = hh_loaders[j]
        else:
            nominal_loader.collate_fn = lambda batch, ix=idx: get_x_i(batch, ix)
            target_loader.collate_fn = lambda batch, ix=idx: get_x_i(batch, ix)

        test_nominal_xi = get_plot_data(nominal_loader)
        test_target_xi = get_plot_data(target_loader)

        save_path = None
        if save:
            save_path = osp.join(tmp_image_dir, f"SMEFT_subdensity_{j}_tmp.png")
            _paths.append(save_path)

        ratio_list = ratio_est if isinstance(ratio_est, (list, tuple)) else [ratio_est]

        utils.plotting.plot_distributions(test_nominal_xi[0], test_target_xi[0],
                                          test_nominal_xi[1], ratio_list, test_target_xi[1],
                                          carl_names=carl_names,
                                          feature_name=feature_labels[j],
                                          nominal_name="SM",
                                          alternate_name=target_name,
                                          percentile_cuts=percentile_cuts,
                                          nominal_mask=np.isfinite, alternate_mask=np.isfinite, carl_mask=np.isfinite,
                                          logscale=False,
                                          typical_ratio=False,
                                          global_name='SMEFT',
                                          show=not save,
                                          saveAs=save_path if save else None,
                                          **kwargs_plot)

        # colelct all saved paths
        if save:
            saved_paths.append(save_path)

    # display saved images in grid (for muon groups multiple images, or jets combined single image)
    if save and show:
        n = len(saved_paths)
        ncols = min(2, n)
        nrows = (n + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
        axes = np.atleast_2d(axes)
        for idx, path in enumerate(saved_paths):
            row = idx // ncols
            col = idx % ncols
            img = mpimg.imread(path)
            axes[row, col].imshow(img)
            axes[row, col].axis('off')
            # axes[row, col].set_title(f"{model_name} - {feature_labels[idx]}", fontsize=16)
        # turn off any unused axes
        for unused in range(len(saved_paths), nrows * ncols):
            r = unused // ncols
            c = unused % ncols
            axes[r, c].axis('off')
        plt.tight_layout()
        plt.show()

        # cleanup
        shutil.rmtree(tmp_image_dir)

    return saved_paths