"""MNIST 0 vs 1 figures.

Main figure: per second-layer sign, the alignment of the filters with their class axis
(cos(w_j, u_+) for v_j > 0, cos(w_j, u_-) for v_j < 0; mean and range over filters of that
sign), and the training loss. Diagnostics (not for the paper): mean alignment with the fixed
axis u_star vs. with the loss-weighted axis u_lam(t), and the stable rank of W.
Input:  artifacts/runs/mnist_p<patch>.npz
Output: results/figures/mnist_{experiment,diagnostics}_p<patch>.{pdf,png}
Usage: python -m analysis.fig_mnist [--patch 14]
"""
import argparse
import logging
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt                                 # noqa: E402
import numpy as np                                              # noqa: E402

from alignment.config import ARTIFACTS, FIGURES, MNIST, MNIST_PATCH_SIZES, MNIST_SEED  # noqa: E402
from alignment.io import cli, load_artifact, setup_logging          # noqa: E402

log = logging.getLogger('analysis.fig_mnist')
COLORS = {'pos': 'tab:green', 'neg': 'tab:red'}


def main_figure(run):
    """Alignment per second-layer sign and the loss curve (three panels)."""
    it, sgn, cref = run['iter'], run['sgn'], run['cref']        # cref: (logs, h, 2)
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    groups = [('pos', sgn > 0, cref[..., 0], r'(a) filters with $v_j>0$: $\cos(w_j,u_+)$'),
              ('neg', sgn < 0, cref[..., 1], r'(b) filters with $v_j<0$: $\cos(w_j,u_-)$')]
    for ax, (key, mask, cos, title) in zip(axes, groups):
        vals = np.where(mask, cos, np.nan)
        ax.plot(it, np.nanmean(vals, 1), color=COLORS[key])
        ax.fill_between(it, np.nanmin(vals, 1), np.nanmax(vals, 1), color=COLORS[key], alpha=0.2)
        ax.set(xlabel='iteration', ylabel='alignment', ylim=(-1.05, 1.05), title=title)
    axes[2].plot(it, run['loss'], color='tab:blue')
    axes[2].set(xlabel='iteration', ylabel='training loss', title='(c) loss curve', ylim=(0, 1.1))
    fig.tight_layout()
    return fig


def diagnostics_figure(run):
    """Fixed-axis vs loss-weighted-axis alignment, and the stable rank of W."""
    it = run['iter']
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.6))
    ax1.plot(it, run['c'].mean(1), label=r'fixed axis $u_\star$')
    ax1.plot(it, run['c_lam'].mean(1), label=r'loss-weighted axis $u_\lambda(t)$')
    ax1.set(xlabel='iteration', ylabel='mean alignment', ylim=(-1.05, 1.05))
    ax1.legend(fontsize=8)
    ax2.plot(it, run['srank'])
    ax2.set(xlabel='iteration', ylabel=r'stable rank $\|W\|_F^2/\|W\|_2^2$')
    fig.tight_layout()
    return fig


def make_figures(run, fig_dir):
    """Both MNIST figures from a loaded MNIST artifact."""
    tag = f"p{run['config']['cfg']['patch']}"
    fig_dir.mkdir(parents=True, exist_ok=True)
    for fig, stem in [(main_figure(run), f'mnist_experiment_{tag}'),
                      (diagnostics_figure(run), f'mnist_diagnostics_{tag}')]:
        for ext in ('pdf', 'png'):
            fig.savefig(fig_dir / f'{stem}.{ext}', dpi=150)
        plt.close(fig)
        log.info('wrote %s/%s.{pdf,png}', fig_dir, stem)


def main(argv=None):
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--patch', type=int, choices=MNIST_PATCH_SIZES, default=MNIST['patch'])
    parser.add_argument('--artifacts', type=Path, default=ARTIFACTS)
    parser.add_argument('--out-dir', type=Path, default=FIGURES)
    args = parser.parse_args(argv)
    setup_logging()
    cfg = dict(MNIST, patch=args.patch)
    run = load_artifact(args.artifacts / f'mnist_p{args.patch}.npz', dict(cfg=cfg, seed=MNIST_SEED))
    make_figures(run, args.out_dir)


if __name__ == '__main__':
    cli(main)
