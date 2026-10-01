# Reflection

## Original state and why it was hard to reproduce

The original analysis (tag `original`) was eight flat Python scripts from a research project on
neuron alignment in two-layer CNNs. The code inside each file was reasonably clean. The
repository as a whole was not reproducible, for these reasons:

- **Two versions of the code were mixed together.** `base_experiment_modified.py` imported
  `common_modified.py`, but `mnist_experiment_modified.py` imported `common.py`.
  `common_modified.train` lacked the `refs` argument that the MNIST script needed, so making the
  imports consistent would have crashed the MNIST script. From the files alone, nobody could tell
  which version had produced the paper figures.
- **Caches could go stale without warning.** `results/base.npz` was reused regardless of the
  configuration or seed. Sweep caches were keyed only by the sweep's name. Changing a setting
  could therefore silently return old results.
- **Settings were changed by hand.** There were commented-out snapshot lines, and `sys.argv`
  overwrote a global config. Importing `common.py` also created output folders as a side effect.
- **Nothing was documented.** There was no environment specification, no README, no stated run
  order and no tests. The key numbers (t₁, t₂, final loss, dead filters) were only printed to
  the terminal.

## Biggest challenges

1. **Choosing the canonical version.** `common.py` turned out to be a superset of
   `common_modified.py`. I kept it, kept the `_modified` experiment scripts, and wrote the
   alignment time as t₁ everywhere.
2. **Restructuring without changing a single number.** In the MNIST run, one random generator
   draws the subsample and then the initialization. Moving data preparation into its own stage
   could have changed the random draws. The processed file therefore stores each digit's images
   in their original order, so the draws are identical to before.
3. **Making outdated results detectable.** Each artifact now stores the configuration that
   produced it. Analysis scripts refuse to use an artifact whose configuration differs from
   `config.py`.

## Improvements with the most impact on reproducibility

- **A regression test** pins the base-run numbers I recorded with the original scripts
  (t₁ = 5780, t₂ = 12180, final loss 0.00445, 10 of 50 filters dead).
- **Configurations are stored in every artifact** and checked when the artifact is loaded.
- **Two documented commands:** `make reproduce` rebuilds everything from raw data in about
  20 minutes, and `make results` rebuilds the figures and tables from the committed checkpoint in
  under a minute.
- **Pinned dependencies** in a venv and an installable `src/` package.

## What I would do differently

I would use git from the first day instead of keeping `_modified` copies. I would give each
stage its own seed, save the configuration with every output, and write the paper figures from
saved artifacts rather than from scripts that retrain.
