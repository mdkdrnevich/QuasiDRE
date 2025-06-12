
import numpy as np
import ot

## To compute a version of the Wasserstein distance which is well-defined for signed measures, we need to decompose the measure

def get_swd_measures(source_data, target_data):
    nom_pos_mask = source_data[1] >= 0
    nom_pos_X = (source_data[0][nom_pos_mask], source_data[1][nom_pos_mask])
    nom_neg_X = (source_data[0][~nom_pos_mask], abs(source_data[1][~nom_pos_mask]))

    target_pos_mask = target_data[1] >= 0
    target_pos_X = (target_data[0][target_pos_mask], target_data[1][target_pos_mask])
    target_neg_X = (target_data[0][~target_pos_mask], abs(target_data[1][~target_pos_mask]))

    source_X = tuple(np.concat([nom_pos_X[i], target_neg_X[i]], axis=0) for i in range(2))
    target_X = tuple(np.concat([target_pos_X[i], nom_neg_X[i]], axis=0) for i in range(2))
    return source_X, target_X


def extended_swd(source_data, target_data, normalized=True, n_projections=50):
    source_X, target_X = get_swd_measures(source_data, target_data)
    if normalized:
        s1, s2 = source_X[1].sum(), target_X[1].sum()
    else:
        s1, s2 = (1, 1)
    dist = ot.sliced_wasserstein_distance(source_X[0], target_X[0], a=source_X[1]/s1, b=target_X[1]/s2, n_projections=n_projections, p=1)
    return dist