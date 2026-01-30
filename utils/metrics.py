import numpy as np
import ot

## To compute a version of the Wasserstein distance which is well-defined for signed measures, we need to decompose the measure

def get_swd_measures(source_data, target_data, normalize=True):
    # First normalize before splitting
    if normalize:
        source_data = (source_data[0], source_data[1]/source_data[1].sum())
        target_data = (target_data[0], target_data[1]/target_data[1].sum())
    nom_pos_mask = source_data[1] >= 0
    nom_pos_X = (source_data[0][nom_pos_mask], source_data[1][nom_pos_mask])
    nom_neg_X = (source_data[0][~nom_pos_mask], abs(source_data[1][~nom_pos_mask]))

    target_pos_mask = target_data[1] >= 0
    target_pos_X = (target_data[0][target_pos_mask], target_data[1][target_pos_mask])
    target_neg_X = (target_data[0][~target_pos_mask], abs(target_data[1][~target_pos_mask]))

    source_X = tuple(np.concat([nom_pos_X[i], target_neg_X[i]], axis=0) for i in range(2))
    target_X = tuple(np.concat([target_pos_X[i], nom_neg_X[i]], axis=0) for i in range(2))
    if normalize:
        source_X = (source_X[0], source_X[1]/source_X[1].sum())
        target_X = (target_X[0], target_X[1]/target_X[1].sum())
    return source_X, target_X


def extended_swd(source_data, target_data, normalize=True, n_projections=50):
    source_X, target_X = get_swd_measures(source_data, target_data, normalize=normalize)
    dist = ot.sliced_wasserstein_distance(source_X[0], target_X[0], a=source_X[1], b=target_X[1], n_projections=n_projections, p=1)
    return dist


def weighted_chi_square_test(x0: np.ndarray, w0: np.ndarray,
                             x1: np.ndarray, w1: np.ndarray,
                             edges: np.ndarray) -> float:
    """Compute a weighted chi-square statistic between two weighted samples.

    The implementation bins the samples according to `edges` and computes the
    chi-square term per bin using weighted variances. Returns the reduced
    chi-square (chi2 / (nbins-2)). Bins with zero variance are skipped.
    """
    sum_w0 = np.sum(w0)
    sum_w1 = np.sum(w1)

    chi_terms = []
    for lower, upper in zip(edges[:-1], edges[1:]):
        mask0 = (x0 > lower) & (x0 < upper)
        mask1 = (x1 > lower) & (x1 < upper)

        w0_i = np.sum(w0[mask0])
        w1_i = np.sum(w1[mask1])

        # Variance proxy from squared weights
        w0_var = np.sum(w0[mask0] ** 2)
        w1_var = np.sum(w1[mask1] ** 2)

        denom = (sum_w0 ** 2) * w1_var + (sum_w1 ** 2) * w0_var
        if denom == 0:
            # Skip ill-defined bins
            continue
        num = (sum_w0 * w1_i - sum_w1 * w0_i) ** 2
        chi_terms.append(num / denom)

    nbins = max(1, len(edges) - 2)
    return float(np.sum(chi_terms) / nbins)


def Tsallis_KL(x0: np.ndarray, w0: np.ndarray,
               x1: np.ndarray, w1: np.ndarray,
               edges: np.ndarray) -> float:
    """Compute a Tsallis-inspired binned KL-like divergence between two weighted samples.

    This function bins the weighted samples and computes sum( a_i^2 / b_i - a_i ) over
    bins, where a and b are the normalized binned weights for the two samples. Bins
    with no support in the reference are skipped.
    """
    w0 = np.asarray(w0, dtype=float).copy()
    w1 = np.asarray(w1, dtype=float).copy()

    # Normalize safely
    s0 = w0.sum()
    s1 = w1.sum()
    if s0 > 0:
        w0 /= s0
    if s1 > 0:
        w1 /= s1

    a_vals = []
    b_vals = []
    for lower, upper in zip(edges[:-1], edges[1:]):
        mask0 = (x0 > lower) & (x0 < upper)
        mask1 = (x1 > lower) & (x1 < upper)
        a_i = np.sum(w0[mask0])
        b_i = np.sum(w1[mask1])
        if np.isclose(b_i, 0.0):
            # Skip bins with no support in the reference
            continue
        a_vals.append(a_i)
        b_vals.append(b_i)

    a_vals = np.array(a_vals)
    b_vals = np.array(b_vals)
    if b_vals.size == 0:
        return 0.0
    return float(np.abs(np.sum((a_vals ** 2) / b_vals - a_vals)))