# Standard Python Modules
import math
import collections
from typing import Iterable, Tuple, Optional, Sequence, Dict, Any

# Pytorch libraries
import torch
from torch.utils.data import DataLoader

# Standard Plotting Toolkit
import matplotlib as mtl
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager

# Notebook and progress reporting
from tqdm import tqdm

# Standard Python data-science packages
import scipy
import numpy as np

# Standard iterative tools for iterators
import itertools

# Small plotting helpers (extracted to separate module)
from .plot_helpers import (
    _safe_normalize,
    write_table,
    ResidualPane,
    ResidualPane_Infill,
)

from .metrics import (
    weighted_chi_square_test,
    Tsallis_KL,
)


# Plotting style from plothist module
from plothist import set_style
from plothist import add_luminosity
set_style('default')

# TROT PATHS
#import sys
#sys.path.append('/afs/desy.de/user/s/sjiggins/HiDA/ML4NW/JupyterHub/ml4nw/utils/TROT')
from .TROT.Generators import real_euc_costs
#from Generators import euc_costs, real_euc_costs  
from .TROT.Tsallis import TROT, q_log
#import TROT, q_log

#import euc_costs, real_euc_costs  

# `weighted_chi_square_test` is implemented in `utils.plot_helpers` and imported above.


# Definition of function for calculating the -ve pdf logical entropy based KL-div
#      a(x)   = p0  ->  {x0,w0}
#      b(x)   = p1  ->  {x1,w1}
# `Tsallis_KL` is implemented in `utils.plot_helpers` and imported above.
    



def get_set_feature(batch_list, set_name: str, set_ix: int, feature_ix: int, features: dict, sort_index: int = 0):
    """Collate function: extract a scalar feature from a set-type feature in each sample.

    Supports both numpy arrays and torch tensors for the event records.
    """
    x_list = []
    w_list = []
    for sample in batch_list:
        for i, feat in enumerate(features):
            if feat != set_name:
                continue
            t = sample[i]
            if isinstance(t, torch.Tensor):
                sorted_t = t[t[:, sort_index].argsort(dim=0, descending=True)]
                try:
                    x_list.append(sorted_t[set_ix, feature_ix].item())
                except Exception:
                    x_list.append(np.nan)
            else:
                idx = np.argsort(t[:, sort_index])[::-1]
                try:
                    x_list.append(t[idx][set_ix, feature_ix])
                except Exception:
                    x_list.append(np.nan)
        w_list.append(sample[-1])

    x_batch = torch.tensor(x_list, dtype=torch.float32).unsqueeze(1)
    w_batch = torch.cat(w_list, dim=0)
    return x_batch, w_batch.unsqueeze(1) 
    

def get_vector_feature(batch_list, name: str, ix: int, features: dict):
    """Collate function: extract a scalar from a vector feature in each sample."""
    x_list = []
    w_list = []
    for sample in batch_list:
        for i, feat in enumerate(features):
            if feat != name:
                continue
            t = sample[i]
            try:
                if isinstance(t, torch.Tensor):
                    x_list.append(t[0, ix].item())
                else:
                    x_list.append(t[0, ix])
            except Exception:
                x_list.append(np.nan)
        w_list.append(sample[-1])

    x_batch = torch.tensor(x_list, dtype=torch.float32).unsqueeze(1)
    w_batch = torch.cat(w_list, dim=0)
    return x_batch, w_batch.unsqueeze(1)


def get_feature_DataLoader(generator, features: dict, feature_name: str, index=None, subfeature_name=None, sort_index: int = 0, sort_feature: Optional[str] = None, batch_size: int = 128, shuffle: bool = False):
    """Return a DataLoader that collates a single feature into (x, w) pairs.

    Raises ValueError for misconfigured arguments.
    """
    features = collections.OrderedDict(sorted(features.items()))
    if sort_feature is not None:
        sort_index = features[feature_name]["subfeatures"].index(sort_feature)

    if features[feature_name]["set"] is True:
        # Expect a 2D index (set_index, feature_index) or provide a subfeature_name
        if subfeature_name is not None and isinstance(index, int):
            index = (index, features[feature_name]["subfeatures"].index(subfeature_name))
        if not (isinstance(index, tuple) and len(index) == 2):
            raise ValueError("If accessing a set feature then provide a 2D index (set_index, feature_index) or a subfeature_name.")
        collate_fn = lambda batch: get_set_feature(batch, feature_name, index[0], index[1], features, sort_index=sort_index)
    else:
        if index is None and subfeature_name is None:
            raise ValueError("If accessing a float/vector feature then provide an index or subfeature_name.")
        if subfeature_name is not None:
            index = features[feature_name]["subfeatures"].index(subfeature_name)
        collate_fn = lambda batch: get_vector_feature(batch, feature_name, index, features)

    return DataLoader(generator, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)


@torch.no_grad()
def get_feature_data(loader: DataLoader) -> Tuple[np.ndarray, np.ndarray]:
    """Extract feature values and weights from a DataLoader with the custom collate_fn."""
    x_list = []
    w_list = []
    for batch in tqdm(loader, total=len(loader)):
        x_list.append(batch[0])
        w_list.append(batch[1])
    x = torch.cat(x_list).numpy().flatten()
    w = torch.cat(w_list).numpy().flatten()
    return x, w


# Idea for later
"""def event_mass(batch_list):
    x_batch_list = []
    w_batch_list = []
    for sample in batch_list:
        _data = sample[0].numpy()
        vec = vector.array(
            {
                "pt": _data[:,0],
                "eta": _data[:,1],
                "phi": _data[:,2],
                "M": _data[:,3]
            }
        )
        total_vec = vec[0]
        for v in vec[1:]:
            total_vec += v
        x_batch_list.append(total_vec.mass)
        w_batch_list.append(sample[2])
    x_batch = torch.tensor(x_batch_list)
    w_batch = torch.cat(w_batch_list, dim=0)
    return x_batch[:, None], w_batch[:, None]"""

# `write_table` is implemented in `utils.plot_helpers` and imported above.

def plot_distributions(nominal_data, alternate_data,
                       nominal_weights, carl_weights, alternate_weights,
                       percentile_cuts=(0.5, 99.5),
                       nominal_mask=np.isfinite, alternate_mask=np.isfinite, 
                       carl_mask=None, nominal_name='Nominal', alternate_name="", 
                       carl_names=None, 
                       feature_name="",
                       typical_ratio = True,
                       ref_name = None,
                       logscale=True, 
                       saveAs=None,
                       nbins = 100,
                       global_name = "",
                       Tsallis_EMD = False,
                       add_table = True,
                       show=True,
                       logging=False,
                       **kwargs):
    """
    Assumes carl weights is an iterable!
    """
    for key,value in kwargs.items():
        try:
            plt.rcParams[key] = value
            if logging:
                print(f'rcParams setting {key}  =  {value}')
        except KeyError:
            pass # not a valid rcParam

    # Font size of legend should adopt size from the rcParams call above so no longer needed here
    #font = font_manager.FontProperties(family='Symbol',
    #                                   style='normal') # size=kwargs['legend.fontsize'])
    if kwargs.get('legend.fontsize'):
        font = font_manager.FontProperties(family='Symbol',
                                       style='normal', size=kwargs['legend.fontsize'])
    else:
        font = font_manager.FontProperties(family='Symbol', style='normal')


    # General histogram settings
    #hist_settings_nom = {'alpha': 0.25, 'color':'slateblue'}
    hist_settings_nom = {'alpha': 0.25, 'color':'tab:blue'}
    #hist_settings_alt = {'alpha': 0.25, 'color':'darkcyan'}
    hist_settings_alt = {'alpha': 0.25, 'color':'tab:orange'}
    hist_settings_CARL = {'histtype':'step', 'linewidth':1.5, 'linestyle':'--', 'alpha': 1}
    hist_settings_CARL_ratio = {'linewidth':1.5, 'linestyle':'--', 'alpha': 1}
    hist_settings_ratio_ref = {'linewidth':1.0, 'linestyle':'--', 'alpha': 1, 'color':'black'}

    # key elements
    label = alternate_name
    legend = label
    legend_title = "Legend"
    column = feature_name
    x_scaler = 1

    # Unbinned data internal assignment
    x0 = nominal_data
    w0 = nominal_weights
    x1 = alternate_data
    w1 = alternate_weights
    w_spec_ref = None
    if ref_name is not None:
        w_spec_ref = carl_weights[carl_names.index(ref_name)]
    
    if nominal_mask is not None or carl_mask is not None:
        athena_mask = np.ones_like(w0, dtype=bool)
        if nominal_mask is not None:
            athena_mask = athena_mask & nominal_mask(nominal_data)
        if carl_mask is not None:
            for carl_w in carl_weights:
                athena_mask = athena_mask & carl_mask(carl_w)        
        x0 = x0[athena_mask]
        w0 = w0[athena_mask]
        w_carl = [w0*wc[athena_mask] for wc in carl_weights]
    if alternate_mask is not None:
        athena_mask = alternate_mask(alternate_data)
        x1 = x1[athena_mask]
        w1 = w1[athena_mask]
    
    if carl_names is None:
        carl_names = ["Nominal*CARL" for _ in range(len(carl_weights))]

    w0 = _safe_normalize(w0)
    w_carl = [_safe_normalize(wc) for wc in w_carl]
    w1 = _safe_normalize(w1)
    if w_spec_ref is not None:
        w_spec_ref = _safe_normalize(w_spec_ref)

    ### Start plotting
    if kwargs.get('figure.figsize'):
        fig = plt.figure(figsize=kwargs['figure.figsize'])
    else:
        fig = plt.figure(figsize=(15, 11))
    gs = fig.add_gridspec(nrows=len(w_carl)+1, ncols=2, 
                          height_ratios=[5] + [2.5 for i in range(len(w_carl))],
                          width_ratios=[12,3],
                          hspace=0.02, wspace=0 )  # Version 2
    axes = gs.subplots(sharex='col')
    axes[0,-1].set_axis_off() # turn off grid lines etc..

    # Calculate binning using percentiles of both samples
    low = min(np.percentile(x0, percentile_cuts[0]), np.percentile(x1, percentile_cuts[0]))
    high = max(np.percentile(x0, percentile_cuts[1]), np.percentile(x1, percentile_cuts[1]))
    if high <= low:
        # fallback to a small symmetric window
        low -= 0.5
        high += 0.5
    binning = np.linspace(low, high, nbins)

    # Statistical Measures
    stat_measures = { r'$\chi^{2}$ Scores' : {}, 
                      r'$D_{q=2}(B || T)$' : {}, }
    if Tsallis_EMD:
        stat_measures['Tsallis EMD'] = {}

    #  EMD parameters
    CostMatrix = real_euc_costs(binning[:-1])
    q = [0.5]
    l = [50]

    # Plot the base and target
    hist_x0, edges_x0, _ = axes[0,0].hist(x0, bins=binning, weights=w0, label=nominal_name, **hist_settings_nom, density=True)
    hist_x1, edges_x1, _ = axes[0,0].hist(x1, bins=binning, weights=w1, label=label, **hist_settings_alt, density=True);

    # =======  Add the closure metrics ==========
    if add_table:
        # Chi^{2}
        chisquares = [weighted_chi_square_test( x1, w1, x0, w0, binning )]
        stat_measures[r'$\chi^{2}$ Scores']['Base / Target:'] = weighted_chi_square_test( x1, w1, x0, w0, binning )
        # Tsallis Relative Entropy
        tsallis_KL = [Tsallis_KL( x1, w1, x0, w0, binning )]
        stat_measures[r'$D_{q=2}(B || T)$']['Base / Target:'] = Tsallis_KL( x1, w1, x0, w0, binning )
        # Tsallis EMD
        if Tsallis_EMD:
            stat_measures['Tsallis EMD']['Base / Target:'] = np.sqrt(np.sum(TROT( q[0], 
                                                                                CostMatrix, 
                                                                                hist_x1, hist_x0,
                                                                                l[0],
                                                                                1E-7)*CostMatrix))

    # Form the CARL histograms
    carl_hists = []
    for i in range(len(w_carl)):
        # - Histograms
        hist, _, _ = axes[0,0].hist(x0, bins=binning, weights=w_carl[i], 
                              label=carl_names[i], **hist_settings_CARL, density=True)
        carl_hists.append(hist)
        
        # - TROT EMD metric
        if Tsallis_EMD and add_table:
            U = TROT( q[0], 
                      CostMatrix, 
                      np.array(hist), np.array(hist_x1),
                      l[0],
                      1E-7)
            #stat_measures['Tsallis EMD'][f'{carl_names[i]} / Target:'] = np.sqrt(np.sum(U*CostMatrix))
            stat_measures['Tsallis EMD'][f'{carl_names[i]}:'] = np.sqrt(np.sum(U*CostMatrix))

    

    # Set the primary axis style - [0,0]
    axes[0,0].set_xlabel('%s'%(column), horizontalalignment='right',x=1)
    axes[0,0].set_ylabel("$\\frac{1}{N} \\cdot \\frac{d \\sigma}{dx}$", horizontalalignment='center',x=1, fontsize=20)
    if logscale is True:
        axes[0,0].set_yscale("log")
    axes[0,0].legend(frameon=False,title = f'{legend_title}', prop=font )
    
    y_min, y_max = axes[0,0].get_ylim()
    axes[0,0].set_ylim([y_min*0.9, y_max*1.35])    

    if add_table:
        # Calculate the closure metrics
        # colors = []
        for idx in range(len(w_carl)):
            # - Closure metrics: Chi^{2}
            chisquares.append(weighted_chi_square_test( x1, w1, x0, w_carl[idx], binning )) 
            #stat_measures[r'$\chi^{2}$ Scores'][f'{carl_names[idx]} / Target:'] = weighted_chi_square_test( x1, w1, x0, w_carl[idx], binning )
            stat_measures[r'$\chi^{2}$ Scores'][f'{carl_names[idx]}:'] = weighted_chi_square_test( x1, w1, x0, w_carl[idx], binning )
            # - Closure metrics: Tsallis Relative Entropy
            tsallis_KL.append( Tsallis_KL( x1, w1, x0, w_carl[idx], binning ) )
            #stat_measures[r'$D_{q=2}(B || T)$'][f'{carl_names[idx]} / Target:'] = Tsallis_KL( x1, w1, x0, w_carl[idx], binning )
            stat_measures[r'$D_{q=2}(B || T)$'][f'{carl_names[idx]}:'] = Tsallis_KL( x1, w1, x0, w_carl[idx], binning )
        
            # - Record the colors for each comparison
            # colors.append(plt.rcParams['axes.prop_cycle'].by_key()['color'][idx])

    # Add the statistical measures to the free top right pane
    if add_table:
        if kwargs.get('table.fontsize'):
            write_table(axes, stat_measures, fontsize=kwargs['table.fontsize'])
        else:
            write_table(axes, stat_measures)

    # ratio plot
    x0_hist, edge0 = np.histogram(x0, bins = binning, weights = w0, density=True)
    x1_hist, edge1 = np.histogram(x1, bins = binning, weights = w1, density=True)
    carl_hists = [np.histogram(x0, bins = binning, weights = wc, density=True) for wc in w_carl]
    try:
        x1_ratio = x0_hist/x1_hist
    except ZeroDivisionError:
        x1_hist[x1_hist == 0] = np.nan
        x1_ratio = x0_hist/x1_hist
    carl_ratio = [carl_hist[0]/x1_hist for carl_hist in carl_hists]
    # Generate reference line
    #   -> Extract the lowest and highest bin edge
    xref= [binning.min(), binning.max()]
    #   -> Now generate the x and y points of the reference line
    yref = [1.0,1.0]

    ## Generate error bands and residue for the reference histogram
    x0_error = []
    x1_error = []
    residue = []
    residue_carl = [[] for _ in w_carl]
    residue_specified_ref = [[] for _ in w_carl]
    if len(binning) > 1:
        width = abs(binning[1] - binning[0] )
        for xbin in binning:
            # Form masks for all event that match condition
            mask0 = (x0 < (xbin + width)) & (x0 > (xbin - width))
            mask1 = (x1 < (xbin + width)) & (x1 > (xbin - width))
            # Form bin error
            binsqrsum_x0      = math.sqrt(np.sum(w0[mask0]**2))
            binsqrsum_x1      = math.sqrt(np.sum(w1[mask1]**2))
            binsqrsum_x0_carl = [math.sqrt(np.sum(wc[mask0]**2)) for wc in w_carl]
            binsqrsum_spec    = math.sqrt(np.sum(w_spec_ref[mask0]**2)) if w_spec_ref is not None else None
            # Form residue
            res_num = np.sum(w1[mask1]) - np.sum(w0[mask0])
            res_denom = math.sqrt(binsqrsum_x0**2 + binsqrsum_x1**2)
            # Form residue (CARL)
            res_num_carl = [np.sum(w1[mask1]) - np.sum(wc[mask0]) for wc in w_carl]
            res_denom_carl = [math.sqrt(binsqrsum_x0_carl_i**2 + binsqrsum_x1**2) for binsqrsum_x0_carl_i in binsqrsum_x0_carl]
            # Form residue (CARL) - specified
            res_num_carl_spec   = None
            res_denom_carl_spec = None
            if ref_name is not None:
                res_num_carl_spec   = [np.sum(w_spec_ref[mask0]) - np.sum(wc[mask0]) for wc in w_carl]
                res_denom_carl_spec = [math.sqrt(binsqrsum_x0_carl_i**2 + binsqrsum_spec**2) for binsqrsum_x0_carl_i in binsqrsum_x0_carl]
            # Form relative error
            try:
                binsqrsum_x0 = binsqrsum_x0/w0[mask0].sum()
            except ZeroDivisionError:
                pass
            try:
                binsqrsum_x1 = binsqrsum_x1/w1[mask1].sum()
            except ZeroDivisionError:
                pass

            # Save residual
            x0_error.append(binsqrsum_x0 if abs(binsqrsum_x0) > 0 else 0.0)
            x1_error.append(binsqrsum_x1 if abs(binsqrsum_x1) > 0 else 0.0)
            residue.append(res_num/res_denom if abs(binsqrsum_x0+binsqrsum_x1) > 0 else 0.0)
            for i in range(len(w_carl)):
                residue_carl[i].append(res_num_carl[i]/res_denom_carl[i] if abs(binsqrsum_x0_carl[i]+binsqrsum_x1) > 0 else 0.0)
            if ref_name is not None:
                for i in range(len(w_carl)):
                    residue_specified_ref[i].append(res_num_carl_spec[i]/res_denom_carl_spec[i] if abs(binsqrsum_x0_carl[i]+binsqrsum_spec) > 0 else 0.0)

    # Convert error lists to numpy arrays
    x0_error = np.array(x0_error)
    x1_error = np.array(x1_error)
    residue  = np.array(residue)
    residue_carl  = np.array(residue_carl)
    residue_specified_ref = np.array(residue_specified_ref)    

    # Automated residual plotting
    #   ->  Calculate the largest range out of all the residuals
    max_resi_range = np.array( [np.abs(arr).max() for arr in residue_carl] ).max() 
    for i in range(len(w_carl)):
        ResidualPane_Infill( i+1, axes=axes, fig=fig,
                             x_ref_centers = xref,  y_ref_residuals = [0.0, 0.0], ref_hist_settings = hist_settings_ratio_ref, 
                             x_carl_centers = carl_hists[i][1], y_carl_residuals = residue_carl[i], carl_hist_settings = hist_settings_CARL_ratio,
                             #alternate__name = alternate_name,
                             comparator_name = carl_names[i],
                             bins = len(edge1),
                             bin_edges = edge1,
                             resi_range = max_resi_range,
                             column=column,
                             color = plt.rcParams['axes.prop_cycle'].by_key()['color'][i])
                            #  color= colors[i])


    # Additinal styling
    add_luminosity(collaboration=f"{global_name}",
                   ax=axes[0,0], fontsize=16, lumi='',
                   preliminary=False)

    if saveAs is not None:
        fig.savefig(saveAs)

    # If requested, prevent the figure from being displayed in IPython notebooks
    if not show:
        plt.close(fig)

    return hist_x0, hist_x1, binning
    # Done!!!

# `ResidualPane` is implemented in `utils.plot_helpers` and imported above.

# `ResidualPane_Infill` is implemented in `utils.plot_helpers` and imported above.


def plot_carl_reweighting(nominal_dataset, alternative_dataset, carl_weights, features, feature_name,
                          index=None, subfeature_name=None, sort_index=0, sort_feature=None, batch_size=128, shuffle=False, # dataloader settings
                          nominal_mask=np.isfinite, alternate_mask=np.isfinite, carl_mask=None, alternate_name="", logscale=True, saveAs=None): # plotting settings
    
    test_nominal_loader = get_feature_DataLoader(nominal_dataset, features, feature_name, index=index, subfeature_name=subfeature_name, sort_index=sort_index, sort_feature=sort_feature, batch_size=batch_size, shuffle=shuffle)
    test_alt_loader = get_feature_DataLoader(alternative_dataset, features, feature_name, index=index, subfeature_name=subfeature_name, sort_index=sort_index, sort_feature=sort_feature, batch_size=batch_size, shuffle=shuffle)

    test_nominal_data = get_feature_data(test_nominal_loader)
    test_alt_data = get_feature_data(test_alt_loader)

    x_title = ""
    if features[feature_name]["set"] is True:
        if subfeature_name is None:
            subfeature_name = features[feature_name]["subfeatures"][index[1]]
            x_title = "{}.{} {}".format(feature_name, subfeature_name, index[0])
        else:
            x_title = "{}.{} {}".format(feature_name, subfeature_name, index)
    else:
        if subfeature_name is None:
            subfeature_name = features[feature_name]["subfeatures"][index]
        x_title = "{}.{}".format(feature_name, subfeature_name)

    plot_distributions(test_nominal_data[0], test_alt_data[0],
                       test_nominal_data[1], carl_weights, test_alt_data[1],
                       feature_name=x_title, alternate_name=alternate_name,
                       nominal_mask=nominal_mask, alternate_mask=alternate_mask, carl_mask=carl_mask, logscale=logscale, saveAs=saveAs)
    # Done!!!
