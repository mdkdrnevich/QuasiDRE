# Standard Python Modules
import math
import collections

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
import uproot
import uproot.exceptions as exceptions
import numpy as np

# Standard iterative tools for iterators
import itertools
import more_itertools

# Plotting style from plothist module
from plothist import set_style
from plothist import add_luminosity
set_style('default')

# TROT PATHS
#import sys
#sys.path.append('/afs/desy.de/user/s/sjiggins/HiDA/ML4NW/JupyterHub/ml4nw/utils/TROT')
from .TROT.Generators import euc_costs, real_euc_costs  
#from Generators import euc_costs, real_euc_costs  
from .TROT.Tsallis import TROT, q_log
#import TROT, q_log

#import euc_costs, real_euc_costs  

def weighted_chi_square_test(x0, w0,
                             x1, w1,
                             edges):
    # Calculate the sum of weights
    sum_w0 = np.sum(w0)
    sum_w1 = np.sum(w1)
    
    # loop through the bin edges
    chi_square_terms = []
    for lower,upper in more_itertools.pairwise(edges):
        # Now calculate bin mask
        x0_i_mask = np.where(  ((x0 < upper) & (x0 > lower)), True, False  )
        x1_i_mask = np.where(  ((x1 < upper) & (x1 > lower)), True, False  )
        
        # Calculate sum of weights
        w0_i = np.sum(w0[x0_i_mask])
        w1_i = np.sum(w1[x1_i_mask])
        
        # Calculate the sum of squared weights ~ variance of bin
        w0_i_std = np.sqrt(np.sum(w0[x0_i_mask]**2))
        w0_i_var = np.sum(w0[x0_i_mask]**2)
        w1_i_std = np.sqrt(np.sum(w1[x1_i_mask]**2))
        w1_i_var = np.sum(w1[x1_i_mask]**2)
        
        # Now calculate the chi-squared term
        numerator = ((sum_w0*w1_i) - (sum_w1*w0_i))**2
        denominator = ((sum_w0**2)*w1_i_var) + ((sum_w1**2)*w0_i_var)
        chi_square_terms.append(numerator/denominator)
        
    # Now calculate and return the total chi-squared value
    return sum(chi_square_terms)/(len(edges)-1-1) # -1 for the edge -> nbins conversion, and then -1 for x^{2} definition


# Definition of function for calculating the -ve pdf logical entropy based KL-div
#      a(x)   = p0  ->  {x0,w0}
#      b(x)   = p1  ->  {x1,w1}
def Tsallis_KL(x0, w0,
               x1, w1,
               edges):

    # Requirement 1:  Ensure that distribution is unit normalised as all pdf values must be 
    #                 sum to unity
    _w0, _w1 = np.copy(w0), np.copy(w1)
    #if not np.isclose(np.sum(_w0), [1.0]):
    #    print(f'The total sum of weights does not conserve probability of 1.0 -> Normalising to unit area')
    _w0 = _w0/np.sum(_w0)
    #if not np.isclose(np.sum(_w1), [1.0]):
    #    print(f'The total sum of weights does not conserve probability of 1.0 -> Normalising to unit area')
    _w1 = _w1/np.sum(_w1)
        
    # Formulate from the bin edges the normalised pdf contributions
    #    -> loop through the bin edges
    _b_factors, _a_factors = [], []
    for lower,upper in more_itertools.pairwise(edges):
        # Now calculate bin mask
        x0_i_mask = np.where(  ((x0 < upper) & (x0 > lower)), True, False  )
        x1_i_mask = np.where(  ((x1 < upper) & (x1 > lower)), True, False  )
        
        # Calculate sum of weights
        #_w0_i_sq = (np.sum(_w0[x0_i_mask]))**2
        _w0_i    = np.sum(_w0[x0_i_mask])
        #_w1_i_sq = (np.sum(_w1[x1_i_mask]))**2
        _w1_i    = np.sum(_w1[x1_i_mask])

        # Check for zero probability
        if np.isclose(_w1_i, [0.0]): 
            # Set to default values to prevent the term contributing
            _w0_i = 0.0
            _w1_i = 1.0

        # Add the support term/factors
        #_a_factors.append(_w0_i_sq)
        _a_factors.append(_w0_i)
        #_b_factors.append(_w1_i_sq)
        _b_factors.append(_w1_i)
    
    _a_factors = np.array(_a_factors)
    _b_factors = np.array(_b_factors)
    
    #return (1 - np.sum(_b_factors))*(np.sum(_a_factors))
    return np.abs(( np.sum( ( (_a_factors**2) / _b_factors ) - _a_factors ) ))
    



def get_set_feature(batch_list, set_name, set_ix, feature_ix, features, sort_index=0):
    x_batch_list = []
    w_batch_list = []
    for sample in batch_list:
        for i, feat in enumerate(features):
            if feat != set_name:
                continue
            t = sample[i]
            # Sort by pT, assuming it's the first column
            t = t[t[:,sort_index].argsort(dim=0, descending=True)]
            try:
                x_batch_list.append(t[set_ix, feature_ix])
            except IndexError:
                x_batch_list.append(np.nan)
        w_batch_list.append(sample[-1])
    x_batch = torch.tensor(x_batch_list)
    w_batch = torch.cat(w_batch_list, dim=0)
    return x_batch[:, None], w_batch[:, None]
    

def get_vector_feature(batch_list, name, ix, features):
    x_batch_list = []
    w_batch_list = []
    for sample in batch_list:
        for i, feat in enumerate(features):
            if feat != name:
                continue
            t = sample[i]
            try:
                x_batch_list.append(t[0, ix])
            except IndexError:
                x_batch_list.append(np.nan)
        w_batch_list.append(sample[-1])
    x_batch = torch.tensor(x_batch_list)
    w_batch = torch.cat(w_batch_list, dim=0)
    return x_batch[:, None], w_batch[:, None]


def get_feature_DataLoader(generator, features, feature_name, index=None, subfeature_name=None, sort_index=0, sort_feature=None, batch_size=128, shuffle=False):
    features = collections.OrderedDict(sorted(features.items()))
    if sort_feature is not None:
        sort_index = features[feature_name]["subfeatures"].index(sort_feature)
    
    loader = DataLoader(generator, batch_size=batch_size, shuffle=shuffle)
    if features[feature_name]["set"] is True:
        try:
            if subfeature_name is not None and type(index) is int:
                index = (index, features[feature_name]["subfeatures"].index(subfeature_name))
            loader.collate_fn = lambda batch: get_set_feature(batch, feature_name, index[0], index[1], features, sort_index=sort_index)
        except TypeError:
            print("ERROR: If accessing a set feature then a 2D index should be provided, i.e. (set_index, feature_index), or index should be the set_index and a subfeature_name should be provided.")
    else:
        if index is None and subfeature_name is None:
            print("ERROR: If accessing a float/vector feature then a single index should be provided or a subfeature_name should be provided.")
        else:
            if subfeature_name is not None:
                index = features[feature_name]["subfeatures"].index(subfeature_name)
            loader.collate_fn = lambda batch: get_vector_feature(batch, feature_name, index, features)
    return loader


@torch.no_grad()
def get_feature_data(loader):
    temp_x = []
    temp_w = []
    t = tqdm(enumerate(loader), total=len(loader))
    for i, batch in t:
        temp_x.append(batch[0])
        temp_w.append(batch[1])
        t.refresh()  # to show immediately the update
    return torch.cat(temp_x).numpy().flatten(), torch.cat(temp_w).numpy().flatten()


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

# Function for writing tables of statistical measures of distance
def write_table(axes,
                stat_measures,
                header_space = 0.05,
                footer_space = 0.05,
                left_margin_space = 0.05,
                stat_entry_sector_indent = 0.1,
                string_value_separation = 0.7,
                fontsize=9):


    # Calculate the stat measure sectors
    stat_sector_size = ( (1.0-header_space)/len(stat_measures) )
    

    # Loop through the dictionary and write the stat measure and the value
    for group_id, (stat_name, instances) in enumerate(stat_measures.items()):
        axes[0,-1].text( x = left_margin_space, 
                         y = 1 - (header_space + (group_id*stat_sector_size)),# + footer_space), 
                         fontsize= fontsize+2,
                         s=stat_name,
                         weight='bold',
                         va='center', ha='left',
                         transform=axes[0,-1].transAxes )
        for entry, (comparison, value) in enumerate(instances.items()): 
            axes[0,-1].text(x = (left_margin_space + stat_entry_sector_indent), 
                            y = 1 - (header_space + (group_id*stat_sector_size) + (entry+1)*(stat_sector_size/(len(instances)+1))),
                            s = comparison,
                            fontsize= fontsize,
                            fontvariant = 'small-caps',
                            va = 'center',
                            ha = 'left',
                            transform=axes[0,-1].transAxes )
            axes[0,-1].text(x = (left_margin_space + stat_entry_sector_indent + string_value_separation),
                            y = 1 - (header_space + (group_id*stat_sector_size) + (entry+1)*(stat_sector_size/(len(instances)+1))),
                            fontsize= fontsize,
                            s = f'{value:.4f}',
                            va = 'center',
                            ha = 'left',
                            transform=axes[0,-1].transAxes )

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
                       **kwargs):
    """
    Assumes carl weights is an iterable!
    """
    for key,value in kwargs.items():
        try:
            plt.rcParams[key] = value
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
        athena_mask = np.zeros(w0.shape) == 0
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

    # Normalize
    w0 /= w0.sum()
    w_carl = [wc / wc.sum() for wc in w_carl]
    w1 /= w1.sum()
    if w_spec_ref is not None:
        w_spec_ref /= w_spec_ref.sum()

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

    # Calculate binning
    binning = np.linspace(min([np.percentile(x0, percentile_cuts[0]), np.percentile(x1, percentile_cuts[0])]),
                          min([np.percentile(x0, percentile_cuts[1]), np.percentile(x1, percentile_cuts[1])]),
                          nbins)
    
    # Statistical Measures
    stat_measures = { r'$\chi^{2}$ Scores' : {}, 
                      r'$D_{q=2}(B || T)$' : {},}
                      #'Tsallis EMD' : {}}
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
        if Tsallis_EMD:
            U = TROT( q[0], 
                      CostMatrix, 
                      np.array(hist), np.array(hist_x1),
                      l[0],
                      1E-7)
            #stat_measures['Tsallis EMD'][f'{carl_names[i]} / Target:'] = np.sqrt(np.sum(U*CostMatrix))
            stat_measures['Tsallis EMD'][f'{carl_names[i]}:'] = np.sqrt(np.sum(U*CostMatrix))

    

    # Set the primary axis style - [0,0]
    axes[0,0].set_xlabel('%s'%(column), horizontalalignment='right',x=1)
    axes[0,0].set_ylabel(r"$\frac{1}{N} \cdot \frac{d \sigma}{dx}$", horizontalalignment='center',x=1, fontsize=20)
    if logscale is True:
        axes[0,0].set_yscale("log")
    axes[0,0].legend(frameon=False,title = f'{legend_title}', prop=font )
    
    y_min, y_max = axes[0,0].get_ylim()
    axes[0,0].set_ylim([y_min*0.9, y_max*1.35])    

    # Calculate the closure metrics
    colors = []
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
        colors.append(plt.rcParams['axes.prop_cycle'].by_key()['color'][idx])

    # Add the statistical measures to the free top right pane
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
                             color= colors[i])


    # Additinal styling
    add_luminosity(collaboration=f"{global_name}",
                   ax=axes[0,0], fontsize=16, lumi='',
                   preliminary=True)

    if saveAs is not None:
        fig.savefig(saveAs)

    return hist_x0, hist_x1, binning
    # Done!!!

# Function for creating the bin by bin ratio pane
def ResidualPane( pane_id, axes,
                  x_ref_centers,  y_ref_residuals, ref_hist_settings, 
                  x_carl_centers, y_carl_residuals, carl_hist_settings,
                  alternate_name = 'Test',
                  bins = 100,
                  bin_edges = None,
                  column='',
                  color='black'):

    ## Plot the residual reference as a straight line
    ref = axes[pane_id,0].step( x_ref_centers, y_ref_residuals, 
                              where="post", 
                              label=alternate_name+" / "+alternate_name, 
                              **ref_hist_settings)
    
    # Plot the residual 
    carl = axes[pane_id,0].step( x_carl_centers, y_carl_residuals, 
                               where="post", 
                               label = '(nominal*CARL) / '+alternate_name, 
                               color=color,
                               **carl_hist_settings)
    axes[pane_id,0].grid(axis='x', color='silver')

    # Error bars/shaded regions for guiding the eyes of the reader
    #   - 1 sigma
    yref_error = np.zeros(bins)
    yref_error_up   = np.full(bins, 1)
    yref_error_down = np.full(bins, -1)
    
    #   - 3 sigma
    yref_3error_up        = np.full(bins, 3)
    yref_3error_up_base   = np.full(bins, 1)
    yref_3error_down      = np.full(bins, -3)
    yref_3error_down_base = np.full(bins, -1)
    
    #   - 5 sigma
    yref_5error_up        = np.full(bins, 5)
    yref_5error_up_base   = np.full(bins, 3)
    yref_5error_down      = np.full(bins, -5)
    yref_5error_down_base = np.full(bins, -3)

    # Make the error bars
    FiveSigma  = axes[pane_id,0].fill_between(bin_edges, yref_5error_up_base, yref_5error_up       , color='lightcoral', alpha=0.2 , label = "5$\sigma$")
    FiveSigma  = axes[pane_id,0].fill_between(bin_edges, yref_5error_down   , yref_5error_down_base, color='lightcoral', alpha=0.2 , label = "5$\sigma$")
    ThreeSigma = axes[pane_id,0].fill_between(bin_edges, yref_3error_up_base, yref_3error_up       , color='bisque'    , alpha=0.3, label = "3$\sigma$")
    ThreeSigma = axes[pane_id,0].fill_between(bin_edges, yref_3error_down   , yref_3error_down_base, color='bisque'    , alpha=0.3, label = "3$\sigma$")
    OneSigma   = axes[pane_id,0].fill_between(bin_edges, yref_error_down    , yref_error_up        , color='olivedrab' , alpha=0.2, label = "1$\sigma$")
    OneSigma   = axes[pane_id,0].fill_between(bin_edges, yref_error_down    , yref_error_up        , color='olivedrab' , alpha=0.2, label = "1$\sigma$")

    # Set labels
    axes[pane_id,0].set_ylabel("Residual",    horizontalalignment='center', x=1)
    axes[pane_id,0].set_xlabel('%s'%(column), horizontalalignment='right',  x=1)
    
    # Set legend
    axes[pane_id,0].legend(frameon=False,
                         ncol=3,
                         handles=[OneSigma,ThreeSigma,FiveSigma], 
                         labels = ["1$\sigma$", "3$\sigma$", "5$\sigma$"])
    
    # Set limits and ticks
    axes[pane_id,0].set_ylim([-7.5, 7.5])
    axes[pane_id,0].set_yticks(np.arange(-7,7,1.0));
    # Done!!!

# Function for creating the bin by bin ratio pane
def ResidualPane_Infill( pane_id, axes, fig,
                         x_ref_centers,  y_ref_residuals, ref_hist_settings, 
                         x_carl_centers, y_carl_residuals, carl_hist_settings,
                         comparator_name = 'Test',
                         bins = 100,
                         resi_profile_bins = 26,
                         bin_edges = None,
                         resi_range = 6,
                         column='',
                         color='black',
                         interp_power=5,
                         color_alpha=0.5):

    ## Plot the residual reference as a straight line
    ref = axes[pane_id,0].step( x_ref_centers, y_ref_residuals, 
                              where="post", 
                              label="Target  / Target", 
                              **ref_hist_settings)
    

    # Plot the residual as a clip mask on what will be a imshow
    t, c, k = scipy.interpolate.splrep(bin_edges, y_carl_residuals, s=0, k=interp_power)
    y_carl_residuals_smoothed = scipy.interpolate.BSpline(t, c, interp_power)

    polygon = axes[pane_id,0].fill_between(bin_edges, np.zeros(bins), y_carl_residuals, color='none', step='mid')
    #polygon = axes[pane_id,0].fill_between(bin_edges, np.zeros(bins), y_carl_residuals_smoothed, color='none')
    verticals = np.vstack( [p.vertices for p in polygon.get_paths()] )
    imshow_extent = [verticals[:,0].min(), verticals[:,0].max(), verticals[:,1].min(), verticals[:,1].max()]
    filling = axes[pane_id,0].imshow(np.abs(y_carl_residuals.reshape(1,-1)), cmap='RdYlGn_r', aspect='auto', alpha = color_alpha,
                                   extent=imshow_extent, vmin=0.0, vmax=5.0)
    filling.set_clip_path(polygon.get_paths()[0], transform=axes[pane_id,0].transData)
    # Now add the smoothed line
    axes[pane_id,0].plot(bin_edges, y_carl_residuals_smoothed(bin_edges), color=color)

    # Set symmetric limits
    max_y_range = np.abs(verticals[:,1].min()) if np.abs(verticals[:,1].min()) > np.abs(verticals[:,1].max()) else np.abs(verticals[:,1].max()) 
    axes[pane_id,0].set_ylim(-1*max_y_range*1.1, max_y_range*1.1)

    # Set labels
    #axes[pane_id,0].set_ylabel(r'$ \frac{b_{i} - t_{i}}{ \sqrt{ \sigma^{2}_{b,i} + \sigma^{2}_{t,i} } }$',    horizontalalignment='center', x=1)
    axes[pane_id,0].set_ylabel(f'{comparator_name} \n vs \n Target', fontsize=int(plt.rcParams['axes.labelsize']*0.7),
                               horizontalalignment='center', x=1)
    axes[pane_id,0].set_xlabel('%s'%(column), horizontalalignment='right',  x=1)

    # Add a general name for the ratio panes
    fig.text(x = 0.008,
             y = 0.35,
             fontsize=plt.rcParams['axes.labelsize'],
             s = r'Pull : $\frac{b_{i} - t_{i}}{ \sqrt{ \sigma^{2}_{b,i} + \sigma^{2}_{t,i} } }$',
             va = 'center',
             ha = 'left',
             #transform=fig.transAxes,
             rotation='vertical')    

    # Style
    axes[pane_id,0].grid(axis='y', linestyle='dashed', which='major', color='darkgrey')
    
    # Set legend
    #  ->  Define patches
    OneSigma   = mtl.patches.Patch(color='olivedrab',  alpha=color_alpha, label = "1$\sigma$")
    ThreeSigma = mtl.patches.Patch(color='bisque',     alpha=color_alpha, label = "3$\sigma$")
    FiveSigma  = mtl.patches.Patch(color='lightcoral', alpha=color_alpha, label = "5$\sigma$")
    if pane_id == 1:  # Only the first one (first ratio pane)
        axes[pane_id,0].legend(frameon=False,
                               ncol=3,
                               handles=[OneSigma,ThreeSigma,FiveSigma])
    
    # Set limits and ticks
    #axes[pane_id,0].set_ylim([-7.5, 7.5])
    #axes[pane_id,0].set_yticks(np.arange(-7,7,1.0));

    # Now plot the projected distribution of residuals
    axes[pane_id,-1].hist(y_carl_residuals, bins=resi_profile_bins, 
                          range = [-1*resi_range, resi_range],
                          histtype='step', color='dimgray')#, orientation='horizontal')
    axes[pane_id,-1].hist( [np.where( np.abs(y_carl_residuals) >= 3, y_carl_residuals, np.nan),
                            np.where( (np.abs(y_carl_residuals) < 3) & (np.abs(y_carl_residuals) >= 1), y_carl_residuals, np.nan),
                            np.where(np.abs(y_carl_residuals) < 1, y_carl_residuals, np.nan)],
                           bins=resi_profile_bins, 
                           range = [-1*resi_range, resi_range],
                           histtype='stepfilled', 
                           stacked=True,
                           color=['lightcoral','bisque', 'olivedrab'],
                           alpha=color_alpha)#, orientation='horizontal')
    # Now set the x/y-axis labels
    axes[pane_id,-1].set_xlabel('Pull', 
                                horizontalalignment='right',x=1)
    axes[pane_id,-1].set_ylabel('Pull Frequency',
                                fontsize=int(plt.rcParams['axes.labelsize']*0.75),
                                horizontalalignment='center', 
                                x=1)

    # Move y-axis to the right
    axes[pane_id,-1].yaxis.set_label_position("right")
    axes[pane_id,-1].yaxis.tick_right()
    #axes[pane_id,-1].set_xlim([-5, 5])
    # Done!!!


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
