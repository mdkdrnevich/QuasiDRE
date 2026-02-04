import numpy as np
import torch

# Import plotting from the package `quasidre` (compat wrapper re-exports `utils.plotting`)
from qdre import plotting


class DummyGenVector:
    def __init__(self, n):
        self.n = n

    def __len__(self):
        return self.n

    def __iter__(self):
        for i in range(self.n):
            # Format: [feature_matrix, weight_tensor]
            # feature_matrix: shape (num_objects, subfeatures)
            yield [np.array([[float(i), 0.1]]), torch.tensor([0.5])]


class DummyGenSet:
    def __init__(self, n):
        self.n = n

    def __len__(self):
        return self.n

    def __iter__(self):
        for i in range(self.n):
            # a set with 3 objects each with (pt, val) columns
            arr = np.array([[i + 2, 0.1], [i + 1, 0.2], [i + 0, 0.3]], dtype=float)
            yield [arr, torch.tensor([0.2])]


def test_get_feature_dataloader_vector_and_extraction():
    features = {'jet': {'set': False, 'subfeatures': ['pt', 'eta']}}
    gen = DummyGenVector(5)
    loader = plotting.get_feature_DataLoader(gen, features, 'jet', index=0, batch_size=2, shuffle=False)

    x, w = plotting.get_feature_data(loader)

    assert x.shape[0] == 5
    assert w.shape[0] == 5
    assert np.allclose(x, np.arange(5))


def test_get_feature_dataloader_set_and_extraction():
    features = {'jet': {'set': True, 'subfeatures': ['pt', 'val']}}
    gen = DummyGenSet(4)
    # get the second-ranked object (set_ix=1), and feature_ix=0 (pt)
    loader = plotting.get_feature_DataLoader(gen, features, 'jet', index=(1, 0), batch_size=2, shuffle=False)

    x, w = plotting.get_feature_data(loader)
    assert x.shape[0] == 4
    assert w.shape[0] == 4
    # since set arrays are created as [[i+2, ...], [i+1, ...], [i+0, ...]] and sorted by pt descending
    # the second ranked object will have pt = i+1
    assert np.allclose(x, np.array([i + 1 for i in range(4)]))
