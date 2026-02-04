import numpy as np

# Import plotting from the package `quasidre` (compat wrapper re-exports `utils.plotting`)
from qdre import plotting


def test_weighted_chi_square_identical():
    x0 = np.array([0.1, 0.2, 0.3, 0.8])
    w0 = np.ones_like(x0)
    x1 = x0.copy()
    w1 = np.ones_like(x1)
    edges = np.linspace(0.0, 1.0, 5)

    chi2 = plotting.weighted_chi_square_test(x0, w0, x1, w1, edges)
    assert np.isfinite(chi2)
    assert abs(chi2) < 1e-8


def test_weighted_chi_square_zero_reference_bins():
    x0 = np.array([0.1, 0.2, 0.3, 0.8])
    w0 = np.ones_like(x0)
    x1 = np.array([0.6, 0.7])
    w1 = np.zeros_like(x1)  # zero weights in reference
    edges = np.linspace(0.0, 1.0, 5)

    chi2 = plotting.weighted_chi_square_test(x0, w0, x1, w1, edges)
    # Should not raise and should be finite (empty/ill-defined bins are skipped)
    assert np.isfinite(chi2)


def test_tsallis_kl_identical_distributions_is_zero():
    x0 = np.linspace(0, 1, 20)
    w0 = np.ones_like(x0)
    x1 = x0.copy()
    w1 = w0.copy()
    edges = np.linspace(0.0, 1.0, 5)

    val = plotting.Tsallis_KL(x0, w0, x1, w1, edges)
    assert np.isfinite(val)
    assert abs(val) < 1e-8


def test_tsallis_kl_skips_zero_reference_bins():
    x0 = np.array([0.1, 0.9])
    w0 = np.array([1.0, 1.0])
    x1 = np.array([0.9])
    w1 = np.array([1.0])
    edges = np.array([0.0, 0.5, 1.0])

    # This places the first point of x0 in the first bin where reference has no support.
    val = plotting.Tsallis_KL(x0, w0, x1, w1, edges)
    assert np.isfinite(val)
    assert val >= 0.0
