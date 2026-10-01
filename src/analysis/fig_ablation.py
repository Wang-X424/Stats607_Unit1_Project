"""Ablation figures: alignment and loss against each swept parameter.

For each sweep, one row of panels, one per landmark iteration of the base run (about t_1/2,
t_1, t_2 and the last iteration); median over seeds with a min-max band.
Primary metric: mean alignment cbar = mean over live filters of cos(w_j, sgn(v_j) u_star).
Secondary metric: training loss.
Input:  artifacts/runs/base.npz, artifacts/runs/sweep_<name>.npz
Output: results/figures/ablation_{alignment,loss}_<name>.{pdf,png}
Usage: python -m analysis.fig_ablation
"""
import argparse
import logging
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt                                 # noqa: E402
import numpy as np                                              # noqa: E402

from alignment.config import ARTIFACTS, BASE, BASE_SEED, FIGURES, SWEEPS  # noqa: E402
from alignment.io import cli, load_artifact, setup_logging          # noqa: E402
from alignment.metrics import landmark_iters                   # noqa: E402
from pipeline.run_sweeps import sweep_config                   # noqa: E402

log = logging.getLogger('analysis.fig_ablation')

METRICS = {'cbar': dict(ylabel=r'mean alignment $\bar c$', yscale='linear', stem='ablation_alignment'),
           'loss': dict(ylabel='training loss', yscale='log', stem='ablation_loss', ylim=(1e-3, 2))}


def sweep_x(name, res):
    """x-axis values of a sweep: the measured noise level for the G sweep, else the grid."""
    return np.median(res['Ghat'], 1) if name == 'G' else res['values']


def sweep_figure(name, res, ks, metric):
    """One row of panels (one per readout iteration in ks) for one sweep and one metric."""
    sw, spec = res['config']['sweep'], METRICS[metric]
    x = sweep_x(name, res)
    fig, row = plt.subplots(1, len(ks), figsize=(3.2 * len(ks), 2.7), sharey=True)
    for ax, k in zip(row, ks):
        Y = res[metric][:, :, np.searchsorted(res['iters'], k)]          # values x seeds
        ax.plot(x, np.median(Y, 1), 'o-', ms=3)
        ax.fill_between(x, Y.min(1), Y.max(1), alpha=0.25)
        ax.set(xscale=sw['xscale'], yscale=spec['yscale'], xlabel=sw['xlabel'], title=f'iteration {k}')
        if sw.get('int_ticks'):
            ax.set_xticks(x, [str(int(v)) for v in x], minor=False)
            ax.set_xticks([], minor=True)
    row[0].set_ylabel(spec['ylabel'])
    if 'ylim' in spec:
        row[0].set_ylim(*spec['ylim'])
    fig.tight_layout()
    return fig


def make_figures(base, sweeps, fig_dir):
    """All ablation figures from the loaded base artifact and a dict {name: sweep artifact}."""
    ks = landmark_iters(base, base['config']['cfg'])
    log.info('readout iterations: %s', ks)
    fig_dir.mkdir(parents=True, exist_ok=True)
    for metric, spec in METRICS.items():
        for name, res in sweeps.items():
            fig = sweep_figure(name, res, ks, metric)
            stem = f"{spec['stem']}_{name}"
            fig.savefig(fig_dir / f'{stem}.pdf')
            fig.savefig(fig_dir / f'{stem}.png', dpi=150)
            plt.close(fig)
            log.info('wrote %s/%s.{pdf,png}', fig_dir, stem)


def load_sweeps(artifact_dir, names=tuple(SWEEPS)):
    """Load and check the sweep artifacts against the current configuration."""
    return {n: load_artifact(artifact_dir / f'sweep_{n}.npz', sweep_config(n)) for n in names}


def main(argv=None):
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--artifacts', type=Path, default=ARTIFACTS)
    parser.add_argument('--out-dir', type=Path, default=FIGURES)
    args = parser.parse_args(argv)
    setup_logging()
    base = load_artifact(args.artifacts / 'base.npz', dict(cfg=BASE, seed=BASE_SEED))
    make_figures(base, load_sweeps(args.artifacts), args.out_dir)


if __name__ == '__main__':
    cli(main)
