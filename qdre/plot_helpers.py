"""Small plotting helper utilities extracted from `plotting.py`.

This module exposes lightweight helpers that are re-used by the higher-level
plotting routines. Keeping them separate makes testing and maintenance easier.
"""
from typing import Tuple
import math

import numpy as np
import matplotlib as mtl
import matplotlib.pyplot as plt
import scipy


def _safe_normalize(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=float)
    s = arr.sum()
    return arr / s if s > 0 else arr


def write_table(axes,
                stat_measures,
                header_space: float = 0.05,
                footer_space: float = 0.05,
                left_margin_space: float = 0.05,
                stat_entry_sector_indent: float = 0.1,
                string_value_separation: float = 0.7,
                fontsize: int = 9):
    """Write key/value statistical measures into the provided axes panel.

    This mirrors the behaviour that was previously embedded in plotting.py.
    """
    stat_sector_size = ((1.0 - header_space) / len(stat_measures))

    for group_id, (stat_name, instances) in enumerate(stat_measures.items()):
        axes[0, -1].text(
            x=left_margin_space,
            y=1 - (header_space + (group_id * stat_sector_size)),  # + footer_space), 
            fontsize=fontsize + 2,
            s=stat_name,
            weight="bold",
            va="center",
            ha="left",
            transform=axes[0, -1].transAxes,
        )
        for entry, (comparison, value) in enumerate(instances.items()):
            axes[0, -1].text(
                x=(left_margin_space + stat_entry_sector_indent),
                y=1
                - (header_space + (group_id * stat_sector_size) + (entry + 1) * (stat_sector_size / (len(instances) + 1))),
                s=comparison,
                fontsize=fontsize,
                fontvariant="small-caps",
                va="center",
                ha="left",
                transform=axes[0, -1].transAxes,
            )
            axes[0, -1].text(
                x=(left_margin_space + stat_entry_sector_indent + string_value_separation),
                y=1
                - (header_space + (group_id * stat_sector_size) + (entry + 1) * (stat_sector_size / (len(instances) + 1))),
                fontsize=fontsize,
                s=f"{value:.4f}",
                va="center",
                ha="left",
                transform=axes[0, -1].transAxes,
            )


def ResidualPane(pane_id, axes,
                  x_ref_centers, y_ref_residuals, ref_hist_settings,
                  x_carl_centers, y_carl_residuals, carl_hist_settings,
                  alternate_name="Test",
                  bins=100,
                  bin_edges=None,
                  column="",
                  color="black"):

    ref = axes[pane_id, 0].step(
        x_ref_centers,
        y_ref_residuals,
        where="post",
        label=alternate_name + " / " + alternate_name,
        **ref_hist_settings,
    )

    carl = axes[pane_id, 0].step(
        x_carl_centers,
        y_carl_residuals,
        where="post",
        label="(nominal*CARL) / " + alternate_name,
        color=color,
        **carl_hist_settings,
    )
    axes[pane_id, 0].grid(axis="x", color="silver")

    # Error bands/shaded regions for guiding the eyes of the reader
    yref_error = np.zeros(bins)
    yref_error_up = np.full(bins, 1)
    yref_error_down = np.full(bins, -1)

    yref_3error_up = np.full(bins, 3)
    yref_3error_up_base = np.full(bins, 1)
    yref_3error_down = np.full(bins, -3)
    yref_3error_down_base = np.full(bins, -1)

    yref_5error_up = np.full(bins, 5)
    yref_5error_up_base = np.full(bins, 3)
    yref_5error_down = np.full(bins, -5)
    yref_5error_down_base = np.full(bins, -3)

    FiveSigma = axes[pane_id, 0].fill_between(
        bin_edges, yref_5error_up_base, yref_5error_up, color="lightcoral", alpha=0.2, label="5$\\sigma$"
    )
    FiveSigma = axes[pane_id, 0].fill_between(
        bin_edges, yref_5error_down, yref_5error_down_base, color="lightcoral", alpha=0.2, label="5$\\sigma$"
    )
    ThreeSigma = axes[pane_id, 0].fill_between(
        bin_edges, yref_3error_up_base, yref_3error_up, color="bisque", alpha=0.3, label="3$\\sigma$"
    )
    ThreeSigma = axes[pane_id, 0].fill_between(
        bin_edges, yref_3error_down, yref_3error_down_base, color="bisque", alpha=0.3, label="3$\\sigma$"
    )
    OneSigma = axes[pane_id, 0].fill_between(
        bin_edges, yref_error_down, yref_error_up, color="olivedrab", alpha=0.2, label="1$\\sigma$"
    )
    OneSigma = axes[pane_id, 0].fill_between(
        bin_edges, yref_error_down, yref_error_up, color="olivedrab", alpha=0.2, label="1$\\sigma$"
    )

    axes[pane_id, 0].set_ylabel("Residual", horizontalalignment="center", x=1)
    axes[pane_id, 0].set_xlabel("%s" % (column), horizontalalignment="right", x=1)

    axes[pane_id, 0].legend(frameon=False, ncol=3, handles=[OneSigma, ThreeSigma, FiveSigma], labels=["1$\\sigma$", "3$\\sigma$", "5$\\sigma$"])

    axes[pane_id, 0].set_ylim([-7.5, 7.5])
    axes[pane_id, 0].set_yticks(np.arange(-7, 7, 1.0))


def ResidualPane_Infill(pane_id, axes, fig,
                         x_ref_centers, y_ref_residuals, ref_hist_settings,
                         x_carl_centers, y_carl_residuals, carl_hist_settings,
                         comparator_name="Test",
                         bins=100,
                         resi_profile_bins=26,
                         bin_edges=None,
                         resi_range=6,
                         column="",
                         color="black",
                         interp_power=5,
                         color_alpha=0.5):

    ref = axes[pane_id, 0].step(x_ref_centers, y_ref_residuals, where="post", label="Target  / Target", **ref_hist_settings)

    t, c, k = scipy.interpolate.splrep(bin_edges, y_carl_residuals, s=0, k=interp_power)
    y_carl_residuals_smoothed = scipy.interpolate.BSpline(t, c, interp_power)

    polygon = axes[pane_id, 0].fill_between(bin_edges, np.zeros(bins), y_carl_residuals, color="none", step="mid")
    verticals = np.vstack([p.vertices for p in polygon.get_paths()])
    imshow_extent = [verticals[:, 0].min(), verticals[:, 0].max(), verticals[:, 1].min(), verticals[:, 1].max()]
    filling = axes[pane_id, 0].imshow(
        np.abs(y_carl_residuals.reshape(1, -1)), cmap="RdYlGn_r", aspect="auto", alpha=color_alpha, extent=imshow_extent, vmin=0.0, vmax=5.0
    )
    filling.set_clip_path(polygon.get_paths()[0], transform=axes[pane_id, 0].transData)
    axes[pane_id, 0].plot(bin_edges, y_carl_residuals_smoothed(bin_edges), color=color)

    max_y_range = np.abs(verticals[:, 1].min()) if np.abs(verticals[:, 1].min()) > np.abs(verticals[:, 1].max()) else np.abs(verticals[:, 1].max())
    axes[pane_id, 0].set_ylim(-1 * max_y_range * 1.1, max_y_range * 1.1)

    axes[pane_id, 0].set_ylabel(
        f"{comparator_name} \n vs \n Target", fontsize=int(plt.rcParams["axes.labelsize"] * 0.7), horizontalalignment="center", x=1
    )
    axes[pane_id, 0].set_xlabel("%s" % (column), horizontalalignment="right", x=1)

    fig.text(
        x=0.008,
        y=0.35,
        fontsize=plt.rcParams["axes.labelsize"],
        s="Pull : $\\frac{b_{i} - t_{i}}{ \\sqrt{ \\sigma^{2}_{b,i} + \\sigma^{2}_{t,i} } }$",
        va="center",
        ha="left",
        rotation="vertical",
    )

    axes[pane_id, 0].grid(axis="y", linestyle="dashed", which="major", color="darkgrey")

    OneSigma = mtl.patches.Patch(color="olivedrab", alpha=color_alpha, label="1$\\sigma$")
    ThreeSigma = mtl.patches.Patch(color="bisque", alpha=color_alpha, label="3$\\sigma$")
    FiveSigma = mtl.patches.Patch(color="lightcoral", alpha=color_alpha, label="5$\\sigma$")
    if pane_id == 1:
        axes[pane_id, 0].legend(frameon=False, ncol=3, handles=[OneSigma, ThreeSigma, FiveSigma])

    axes[pane_id, -1].hist(y_carl_residuals, bins=resi_profile_bins, range=[-1 * resi_range, resi_range], histtype="step", color="dimgray")
    axes[pane_id, -1].hist(
        [
            np.where(np.abs(y_carl_residuals) >= 3, y_carl_residuals, np.nan),
            np.where((np.abs(y_carl_residuals) < 3) & (np.abs(y_carl_residuals) >= 1), y_carl_residuals, np.nan),
            np.where(np.abs(y_carl_residuals) < 1, y_carl_residuals, np.nan),
        ],
        bins=resi_profile_bins,
        range=[-1 * resi_range, resi_range],
        histtype="stepfilled",
        stacked=True,
        color=["lightcoral", "bisque", "olivedrab"],
        alpha=color_alpha,
    )

    axes[pane_id, -1].set_xlabel("Pull", horizontalalignment="right", x=1)
    axes[pane_id, -1].set_ylabel("Pull Frequency", fontsize=int(plt.rcParams["axes.labelsize"] * 0.75), horizontalalignment="center", x=1)

    axes[pane_id, -1].yaxis.set_label_position("right")
    axes[pane_id, -1].yaxis.tick_right()
