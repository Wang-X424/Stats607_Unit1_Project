"""Base experiment figures: filter snapshots in the signal plane and the loss curve.

Five snapshots of the filters projected onto the signal plane (initialization, end of
alignment t_1, end of fitting, and the two midpoints), in two versions: filters normalized to
unit norm before projection (directions only) and raw projections (norms visible); and the
loss curve with the measured t_1 and t_2.
Input:  artifacts/runs/base.npz
Output: results/figures/base_experiment_{normalized,raw,loss}.{pdf,png}
Usage: python -m analysis.fig_base
"""
import argparse
import logging
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt                                 # noqa: E402
import numpy as np                                              # noqa: E402

from alignment.config import ARTIFACTS, BASE, BASE_SEED, FIGURES  # noqa: E402
from alignment.io import cli, load_artifact, setup_logging          # noqa: E402
from alignment.metrics import base_snapshots                   # noqa: E402

log = logging.getLogger('analysis.fig_base')

COLORS = {'pos_data': 'tab:blue', 'neg_data': 'tab:orange',
          'pos_filter': 'tab:green', 'neg_filter': 'tab:red', 'dead': 'lightgrey'}
# filled markers with a thin black edge so that filters stay visible on top of the data points
FILTER_MARKER = {True: dict(marker='o', ms=5, mec='k', mew=0.4, ls=''),     # v_j > 0
                 False: dict(marker='X', ms=6, mec='k', mew=0.4, ls='')}    # v_j < 0
PAPER_WIDTH = 6.9   # inches; close to the final single-column width so LaTeX does not shrink text


def save(fig, fig_dir, stem, dpi=300):
    """Save a figure as PDF and PNG in fig_dir and close it."""
    fig_dir.mkdir(parents=True, exist_ok=True)
    for ext in ('pdf', 'png'):
        fig.savefig(fig_dir / f'{stem}.{ext}', dpi=dpi, bbox_inches='tight', pad_inches=0.02)
    plt.close(fig)
    log.info('wrote %s/%s.{pdf,png}', fig_dir, stem)


def draw_plane(ax, run, i, normalized, lim):
    """Draw the signal plane at log index i.

    Signal patches on the unit circle, the class axes u_+ and u_- (normalized sums of the
    positive / negative signal patches, drawn with unit length), and one marker per filter at
    P_sig(w_j / ||w_j||) if normalized else P_sig(w_j). Dots for v_j > 0, crosses for v_j < 0;
    filters that activate no signal patch (frozen for good) in grey.
    """
    s, y = run['signals'], run['y']
    ax.plot(*s[y > 0].T, 'o', ms=5, mfc='none', color=COLORS['pos_data'], label='positive signal patch')
    ax.plot(*s[y < 0].T, 'x', ms=5, color=COLORS['neg_data'], label='negative signal patch')
    for cls, key, label in [(y > 0, 'pos_filter', '$u_+$'), (y < 0, 'neg_filter', '$u_-$')]:
        u = s[cls].sum(0) / np.linalg.norm(s[cls].sum(0))
        ax.plot([0, u[0]], [0, u[1]], '-', lw=1.5, color=COLORS[key], label=label)
    proj, sgn, alive = run['proj'][i], run['sgn'][i], run['alive'][-1]
    if normalized:
        proj = proj / run['norm'][i][:, None]
    for p, s_, a in zip(proj, sgn, alive):
        color = COLORS['dead'] if not a else COLORS['pos_filter'] if s_ > 0 else COLORS['neg_filter']
        ax.plot(p[0], p[1], **FILTER_MARKER[s_ > 0], color=color)
    ax.set(xlim=(-lim, lim), ylim=(-lim, lim), aspect='equal')


def snapshot_figure(run, snapshots, normalized):
    """One row of signal-plane panels, one per snapshot iteration, with a shared legend."""
    idx = lambda k: int(np.searchsorted(run['iter'], k))        # noqa: E731
    lim = 1.3 if normalized else max(1.3, 1.1 * np.abs(run['proj'][idx(snapshots[-1])]).max())
    fig, axes = plt.subplots(1, len(snapshots), figsize=(PAPER_WIDTH, 2.15),
                             sharex=True, sharey=True, gridspec_kw={'wspace': 0.08})
    for col, (ax, k, letter) in enumerate(zip(axes, snapshots, 'abcde')):
        draw_plane(ax, run, idx(k), normalized, lim)
        ax.set_title(rf'({letter}) $iter={k}$', fontsize=8, pad=2)
        if normalized:
            ax.set_xticks([-1.0, 0.0, 1.0])
            ax.set_yticks([-1.0, 0.0, 1.0])
        ax.tick_params(axis='both', which='major', labelsize=7, pad=1.5)
        if col > 0:
            ax.tick_params(labelleft=False)
    for key, positive, label in [('pos_filter', True, r'filter, $v_j>0$'),
                                 ('neg_filter', False, r'filter, $v_j<0$'),
                                 ('dead', False, 'filter, no signal patch active')]:
        axes[0].plot([], [], **FILTER_MARKER[positive], color=COLORS[key], label=label)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 0.01), ncol=4,
               fontsize=7.2, frameon=False, handlelength=1.4, handletextpad=0.4, columnspacing=1.0)
    fig.subplots_adjust(left=0.055, right=0.995, top=0.89, bottom=0.28)
    return fig


def loss_figure(run, t_1, t_2):
    """Training loss against iteration with the measured t_1 and t_2 marked."""
    fig, ax = plt.subplots(figsize=(4.8, 2.8))
    ax.plot(run['iter'], run['loss'], color='tab:blue')
    for k, name in [(t_1, r'$\hat t_1$'), (t_2, r'$\hat t_2$')]:
        ax.axvline(k, color='k', ls='--', lw=0.8)
        ax.annotate(f'{name} = {k}', (k, 1.02), rotation=90, va='bottom', ha='right', fontsize=9)
    ax.set(xlabel='iteration', ylabel='training loss', ylim=(0, 1.35))
    ax.tick_params(axis='both', labelsize=8)
    ax.xaxis.label.set_size(9)
    ax.yaxis.label.set_size(9)
    fig.tight_layout()
    return fig


def make_figures(run, fig_dir):
    """All base-experiment figures from a loaded base artifact."""
    marks = base_snapshots(run, run['config']['cfg'])
    for normalized, tag in [(True, 'normalized'), (False, 'raw')]:
        save(snapshot_figure(run, marks['snapshots'], normalized), fig_dir, f'base_experiment_{tag}')
    save(loss_figure(run, marks['t_1'], marks['t_2']), fig_dir, 'base_experiment_loss')


def main(argv=None):
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--artifacts', type=Path, default=ARTIFACTS)
    parser.add_argument('--out-dir', type=Path, default=FIGURES)
    args = parser.parse_args(argv)
    setup_logging()
    run = load_artifact(args.artifacts / 'base.npz', dict(cfg=BASE, seed=BASE_SEED))
    make_figures(run, args.out_dir)


if __name__ == '__main__':
    cli(main)
