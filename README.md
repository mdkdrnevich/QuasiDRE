# QuasiDRE
Density-ratio estimation tools and experiments for quasi-probabilistic distributions.

---

## Quick overview 💡
**QuasiDRE** is a collection of notebooks, training utilities, evaluation code and tests for density-ratio estimation methods (SMMs, CARL, REVERT, etc.). The repository contains both physics-inspired experiments (`SMEFT/`) and smaller toy-model experiments (`ToyModels/`) along with shared utilities under the `qdre/` package (core helpers for preprocessing, training, plotting and metrics).

### Associated publication 📚
This codebase was developed to accompany the paper "Quasi-probabilistic Density-Ratio Estimation" (arXiv:2512.19913). The repository contains the implementation, notebooks, and experiments used in the paper — including data preprocessing, model training (SMMs, CARL, REVERT), evaluation code, and plotting utilities — to reproduce the figures and numerical results reported in the manuscript.

Please cite the paper when using this code: https://arxiv.org/abs/2512.19913

**Suggested citation (BibTeX):**

```bibtex
@article{Drnevich2025Quasiprobabilistic,
  author = {Drnevich, Matthew and Jiggins, Stephen and Cranmer, Kyle},
  title = {Quasiprobabilistic Density Ratio Estimation with a Reverse Engineered Classification Loss Function},
  journal = {arXiv preprint},
  eprint = {2512.19913},
  archivePrefix = {arXiv},
  year = {2025},
  doi = {10.48550/arXiv.2512.19913},
  url = {https://arxiv.org/abs/2512.19913},
}
```


---

## Repository layout 🔧
Top-level structure (short):

- `examples/SMEFT/` - Notebooks and data for SMEFT-based experiments; includes preprocessing, training, evaluation and plotting notebooks and the `data/` and `models/` subfolders.
- `examples/ToyModels/` - Toy model data, generation notebooks and experiments used for quick prototyping.
- `qdre/` - Core Python package used across experiments with the following submodules:
  - `qdre/plotting.py` — plotting helpers, feature extraction and distribution/evaluation utilities used by tests and notebooks
  - `qdre/metrics.py` — statistical measures and tests (weighted chi-square, Tsallis KL, SWD)
  - `qdre/preprocessing.py` — data collation / preprocessing helpers used by training/evaluation loops
  - `qdre/train.py` — training and evaluation loops, loss helpers (Pare, REVERT) and serialization helpers
  - `qdre/models.py` — small model definitions used in the notebooks
- `tests/` - Pytest unit tests that exercise plotting helpers and metrics (`tests/test_dataloader.py`, `tests/test_stats.py`)
- `README.md` - this file


---

## Quickstart / Development 🛠️

Recommended environment:
- Python 3.8+ (3.9/3.10 are known to work)
- Core packages: `numpy`, `torch` (PyTorch), `matplotlib`, `tqdm`, `pytest`.
- Optional: `jupyterlab` or `notebook` for running the notebooks interactively; `plothist` is used for plotting style in `utils/plotting.py`.

Installation example (pip):

```bash
# install core requirements (adjust for CPU/GPU PyTorch build)
python -m pip install numpy matplotlib tqdm pytest jupyter
# For PyTorch, follow official instructions: https://pytorch.org/
# Optional: if you use plotting style provided by `plothist` install it too
python -m pip install plothist
```

Note: A `requirements.txt` file is included with pinned dependencies for convenience. The project also includes a `pyproject.toml` for packaging metadata.


---

## Running tests ✅
Run the unit tests with pytest:

```bash
python -m pytest -q
```

The tests under `tests/` import directly from `qdre.plotting` to avoid package-level side-effects and exercise the feature extraction and implemented metrics.


---

## Notebooks & experiments 📓

- `SMEFT/` contains the main analysis notebooks used for data processing, training different classifiers and evaluating them using the SMEFT dataset (see below).
- `ToyModels/` is a lighter-weight playground for synthetic data generation, training and evaluation (good for fast iteration).

Open any notebook with Jupyter/VSCode and run the cells to reproduce or explore experiments.


---

## Code notes & how to use the utilities 📚

- Use `qdre.train` for training/testing loops (functions `train()` and `test()` expect a `torch.utils.data.DataLoader` yielding `(x, y, w)`).
- Use `qdre.plotting` for extracting scalar features from event records (`get_feature_DataLoader`, `get_feature_data`) and for plotting distributions and closure metrics used across notebooks.
- Preprocessing helpers in `qdre.preprocessing` supply collate functions expected by the trainers and evaluation code.

Example usage snippet:

```python
from qdre import train, plotting
# prepare a DataLoader that yields (x, y, w)
# call train.train(model, optimizer, loader, ...)
# or: from qdre import plotting; plotting.get_feature_DataLoader(...)
```


---

## Data & results 📁

- By default, input datasets (SMEFT / ToyModels) are stored under `SMEFT/data/` and `ToyModels/data/` as `.npy` arrays for fast loading. Some datasets (if present) include a `.root` source.
- The SMEFT dataset used in the paper ("Neural Quasiprobabilistic Likelihood Ratio Estimation Dataset") is available on Zenodo: https://doi.org/10.5281/zenodo.15102316. This archive contains the full SMEFT files (e.g. `SMEFT_EFT_combined_tuple.root`, `SMEFT_SM_combined_tuple.root`) and the ToyModels data used for the experiments. Please cite the dataset when using it:

  [1]S. Jiggins and M. Drnevich, “Neural Quasiprobabilistic Likelihood Ratio Estimation Dataset”. Zenodo, Mar. 28, 2025. doi: 10.5281/zenodo.15102316.

- By default, trained models and evaluation outputs are stored in `models/` and `results_SWD/` (for Sliced-Wasserstein result distributions).


---

## Contributing & Issues 🙌

- Open issues for bugs or feature requests. Contributions via pull requests are welcome — please include tests where reasonable.

---

## Contact & Notes ✉️

If you need help understanding any piece of the code base, start with the notebooks in `SMEFT/` and `ToyModels/` (they provide runnable examples).

---

Thank you for exploring QuasiDRE! 🚀
