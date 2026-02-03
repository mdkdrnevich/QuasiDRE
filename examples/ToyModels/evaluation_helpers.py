import torch
import numpy as np
import tempfile
import os.path as osp
import shutil
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from qdre.plotting import get_x_i, get_plot_data, plot_distributions


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


def plot_closure(ratio_est,
                 nominal_loader,
                 target_loader,
                 nominal_name,
                 target_name,
                 percentile_cuts=(0.0, 100.0),
                 carl_names=None,
                 kwargs_plot=None,
                 figsize=(16,12),
                 show=True,
                 save=True):
    """
    Plot CARL reweighting closure for a subdensity (wraps the code from the previous cell).
    - percentile_cuts: tuple for plot_distributions
    - carl_names: list of names for CARL curves (defaults to ["<name> CARL"])
    - kwargs_plot: dict of plotting kwargs (font sizes etc.), defaults to the notebook kwargs
    - show: whether to display images inline
    - save: whether to save temporary images (if False will just call plot_distributions)
    Returns list of saved image paths (empty list if save=False).
    """
    if carl_names is None:
        carl_names = [f"CARL"]

    if kwargs_plot is None:
        kwargs_plot = {'legend.title_fontsize': 18, 'legend.fontsize': 14,
                        'font.size': 14, 'axes.titlesize':20, 'axes.labelsize':16,'figure.titlesize':20, 'ytick.labelsize':12}

    saved_paths = []

    # mapping for groups to base index + number of plots
    n_plots = 3
    feature_labels = ["X Coordinate", "Y Coordinate", "Radial Coordinate ($R = \\sqrt{X^2+Y^2}$)"]

    # temporary dir for saved images
    if save:
        tmp_image_dir = tempfile.mkdtemp()
        _paths = []

    for j in range(n_plots):
        if j == 2:  # radial coordinate
            nominal_loader.collate_fn = lambda batch: get_r(batch)
            target_loader.collate_fn = lambda batch: get_r(batch)
        else:
            nominal_loader.collate_fn = lambda batch, ix=j: get_x_i(batch, ix)
            target_loader.collate_fn = lambda batch, ix=j: get_x_i(batch, ix)

        test_nominal_xi = get_plot_data(nominal_loader)
        test_target_xi = get_plot_data(target_loader)

        save_path = None
        if save:
            save_path = osp.join(tmp_image_dir, f"ToyModel_subdensity_{j}_tmp.png")
            _paths.append(save_path)

        ratio_list = ratio_est if isinstance(ratio_est, (list, tuple)) else [ratio_est]

        assert len(ratio_list) == len(carl_names), "Length of ratio_list and carl_names must match."

        plot_distributions(test_nominal_xi[0], test_target_xi[0],
                            test_nominal_xi[1], ratio_list, test_target_xi[1],
                            carl_names=carl_names,
                            feature_name=feature_labels[j],
                            nominal_name=nominal_name,
                            alternate_name=target_name,
                            percentile_cuts=percentile_cuts,
                            nominal_mask=np.isfinite, alternate_mask=np.isfinite, carl_mask=np.isfinite,
                            logscale=False,
                            typical_ratio=False,
                            global_name='Gaussian Mixture',
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