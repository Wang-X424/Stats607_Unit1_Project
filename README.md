# Neuron alignment in a two-layer CNN

Simulations and an MNIST experiment on how the first-layer filters of a two-layer CNN,
trained by full-batch gradient descent from a small initialization, first **align** with the
class axes of the data and only then **fit** it.

The model is `f(x) = Σ_j v_j Σ_p ReLU(⟨w_j, x^(p)⟩)`, and both layers are trained on the mean
exponential loss. The synthetic data follow a two-cone signal–noise patch model: patch 0 is a
signal patch in the plane span{e₁, e₂} within angle θ\*/2 of its class axis u₊ or u₋, and the
remaining P−1 patches are Gaussian noise in the other D−2 coordinates. Two landmark times are
measured from each training log:

- **t₁ (alignment):** the first iteration at which every live filter points inside the cone of its class;
- **t₂ (fitting):** the first iteration at which Q₊ and Q₋ both exceed b₀.

## Principal outputs

| Output | File(s) in `results/` | Produced by |
|---|---|---|
| Filter snapshots in the signal plane (normalized and raw) and the loss curve with t₁, t₂ | `figures/base_experiment_{normalized,raw,loss}.{pdf,png}` | `analysis.fig_base` |
| Ablations: alignment c̄ and loss vs ε, P, Ĝ, θ\*, read out at ≈t₁/2, t₁, t₂ and the end | `figures/ablation_{alignment,loss}_{eps,P,G,theta}.{pdf,png}` | `analysis.fig_ablation` |
| MNIST 0 vs 1: alignment per second-layer sign, and the loss; diagnostics | `figures/mnist_{experiment,diagnostics}_p14.{pdf,png}` | `analysis.fig_mnist` |
| Summary tables (landmarks, sweep medians, MNIST statistics) | `tables/{base_summary,sweep_summary,mnist_summary_p14}.csv` | `analysis.make_tables` |

## Data

- **Synthetic data** are generated from the seeds in `src/alignment/config.py`. There is nothing to download.
- **MNIST** (training set, digits 0 and 1) is public. `make data` downloads
  `train-images-idx3-ubyte.gz` and `train-labels-idx1-ubyte.gz` into `data/raw/`, checks their
  MD5 checksums, and writes the digit-0/1 images to `data/processed/mnist01.npz`. The raw files
  are not committed. If no mirror is reachable, download the two files by hand
  (for example from <https://ossci-datasets.s3.amazonaws.com/mnist/>) and put them in `data/raw/`.

## Setup

Requirements: Python ≥ 3.10 (tested with 3.10.13 on macOS), `make`, and `git`. No other
system libraries are needed. On macOS, `make` comes with the Xcode command line tools
(`xcode-select --install`).

```bash
git clone <https://github.com/Wang-X424/Stats607_Unit1_Project>
cd <Stats607_Unit1_Project>
make env                     # creates .venv, installs requirements.txt and this package
# equivalently:
#   python3 -m venv .venv
#   .venv/bin/python -m pip install -r requirements.txt
#   .venv/bin/python -m pip install --no-deps -e .
```

If `python3` is not the interpreter you want, pass one explicitly: `make env PYTHON=python3.10`.

## Reproducing the results

There are two paths. Both write to `results/`.

| Command | What it runs | Time |
|---|---|---|
| `make results` | Figures and tables from the **committed checkpoint** in `artifacts/runs/`. No training, no download. | < 1 min |
| `make reproduce` | **Everything from raw data:** downloads MNIST, runs the base run, the sweeps (27 settings × 5 seeds) and the MNIST run, then `make results`. | ≈ 20–30 min |

Individual stages: `make data`, `make base`, `make sweeps`, `make mnist`
(`make mnist results PATCH=7` for another patch size in {4, 7, 14, 28}). `make help` lists all targets.
Every stage logs its progress to the terminal. All randomness is seeded (`BASE_SEED`,
`SWEEP_SEEDS` and `MNIST_SEED` in `src/alignment/config.py`).

Every artifact stores the configuration that produced it. If you change a setting in
`config.py` and run `make results` without rerunning the pipeline, it stops with an error
instead of silently using stale results.

## Tests

```bash
make test         # all tests, about 15 s
make test-fast    # skips the full base-run regression test
```

- `tests/test_data.py`: **data validation.** Shapes, balanced labels, signal and noise supports,
  every signal patch inside its class cone, and MNIST patches tiling the centred image.
- `tests/test_model.py`: **function correctness.** Analytic gradients against finite differences,
  angle wrap-around, iteration rounding, and deterministic training.
- `tests/test_pipeline.py`: **pipeline integrity.** Every stage runs end to end on a small
  configuration and writes the expected files. Missing or stale artifacts fail with clear errors.
  The full base run reproduces the numbers obtained with the original scripts
  (t₁ = 5780, t₂ = 12180, final loss 0.00445, 10 of 50 filters dead).

## Repository structure

```
Stats607_Unit1_Project/
├── .gitignore
├── Makefile                        # env / reproduce / pipeline / results / test / clean
├── README.md
├── pyproject.toml                  # src layout; installed with pip install -e .
├── requirements.txt                # pinned numpy, matplotlib, pytest
│
├── data/
│   ├── raw/                        # MNIST .gz files, downloaded and MD5-checked
│   └── processed/
│       └── mnist01.npz             # uint8 images of digits 1 and 0
│
├── src/
│   ├── alignment/                  # core library, no side effects on import
│   │   ├── config.py               # BASE, SWEEPS, MNIST configs, seeds, paths
│   │   ├── model.py                # data model, forward/backward, gradient descent training
│   │   ├── metrics.py              # landmark times t_1, t_2; class angles; snapshot iterations
│   │   ├── mnist.py                # 
│   │   └── io.py                   # artifacts saved with their config
│   ├── pipeline/                   # populates data/processed/ and artifacts/
│   │   ├── prepare_mnist.py        # data/raw -> data/processed/mnist01.npz
│   │   ├── run_base.py             # -> artifacts/runs/base.npz
│   │   ├── run_sweeps.py           # -> artifacts/runs/sweep_{eps,P,G,theta}.npz
│   │   └── run_mnist.py            # -> artifacts/runs/mnist_p14.npz
│   └── analysis/                   # populates results/ from artifacts/ only
│       ├── fig_base.py             # filter snapshots in the signal plane, loss curve
│       ├── fig_ablation.py         # alignment and loss vs. each swept parameter
│       ├── fig_mnist.py            # MNIST alignment per second-layer sign, diagnostics
│       └── make_tables.py          # CSV summary tables
│
├── artifacts/
│   └── runs/                       # training logs (.npz): the committed checkpoint behind `make results`
│
├── results/                        # regenerated by `make results` (not committed)
│   ├── figures/                    # base_experiment_*, ablation_{alignment,loss}_*, mnist_*_p14 (.pdf, .png)
│   └── tables/                     # base_summary.csv, sweep_summary.csv, mnist_summary_p14.csv
│
├── tests/                          # pytest suite, run with `make test`
│   ├── conftest.py                 # small configuration for fast tests
│   ├── test_data.py                # data validation
│   ├── test_model.py               # function correctness: gradient check, helpers, divergence detection
│   └── test_pipeline.py            # pipeline integrity, plus regression against the original results
│
└── docs/
    └── reflection.md
```

## Provenance

The original, unstructured scripts are preserved at the git tag `original`.
