"""Summary tables of the base run, the sweeps and the MNIST run.

Input:  artifacts/runs/{base,sweep_<name>,mnist_p<patch>}.npz
Output: results/tables/base_summary.csv, sweep_summary.csv, mnist_summary_p<patch>.csv
Usage: python -m analysis.make_tables [--patch 14]
"""
import argparse
import csv
import logging
from pathlib import Path

import numpy as np

from alignment.config import ARTIFACTS, BASE, BASE_SEED, MNIST, MNIST_PATCH_SIZES, MNIST_SEED, TABLES
from alignment.io import cli, load_artifact, setup_logging
from alignment.metrics import base_snapshots, class_angle, first_iter, landmark_iters

log = logging.getLogger('analysis.make_tables')


def write_csv(path, rows):
    """Write a list of dicts (same keys) to a CSV file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    log.info('wrote %s (%d rows)', path, len(rows))


def base_summary(run):
    """Landmark times and end-of-training statistics of the base run (one row)."""
    cfg = run['config']['cfg']
    marks = base_snapshots(run, cfg)
    alive = run['alive'][-1]
    return dict(t_1=marks['t_1'], t_S=first_iter(run, run['fracS'] >= alive.mean(), ''),
                t_2=marks['t_2'], t_loss_005=marks['t_end'],
                snapshots=' '.join(map(str, marks['snapshots'])),
                final_loss=float(f"{run['loss'][-1]:.3g}"),
                dead_filters=int((~alive).sum()), filters=len(alive),
                max_final_angle_deg=round(float(class_angle(run, cfg)[-1, alive].max()), 1))


def sweep_summary(base, sweeps):
    """Median over seeds of cbar at each readout iteration, noise level and dead fraction."""
    ks = landmark_iters(base, base['config']['cfg'])
    rows = []
    for name, res in sweeps.items():
        key = res['config']['sweep']['key']
        for i, val in enumerate(res['values']):
            row = dict(sweep=name, parameter=key, value=float(val),
                       Ghat_median=round(float(np.median(res['Ghat'][i])), 4),
                       dead_fraction_median=round(float(np.median(res['dead'][i])), 4))
            for k in ks:
                j = np.searchsorted(res['iters'], k)
                row[f'cbar_iter{k}'] = round(float(np.median(res['cbar'][i, :, j])), 4)
            rows.append(row)
    return rows


def mnist_summary(run):
    """Shape, class-axis geometry, alignment peak and end values of the MNIST run (one row)."""
    it, c = run['iter'], run['c'].mean(1)
    peak = int(np.argmax(c))
    n, P, D = map(int, run['shape'])
    r = lambda x: round(float(x), 3)                            # noqa: E731
    return dict(patch=run['config']['cfg']['patch'], n=n, P=P, D=D,
                cos_u_plus_u_minus=r(run['u_plus'] @ run['u_minus']),
                cbar_peak=r(c[peak]), cbar_peak_iter=int(it[peak]), loss_at_peak=r(run['loss'][peak]),
                cbar_final=r(c[-1]), c_lam_final=r(run['c_lam'][-1].mean()),
                srank_peak=round(float(run['srank'][peak]), 2), srank_final=round(float(run['srank'][-1]), 2),
                t_loss_half=first_iter(run, run['loss'] < 0.5 * run['loss'][0], 'never'),
                final_loss=float(f"{run['loss'][-1]:.3g}"))


def main(argv=None):
    """Command-line entry point."""
    from analysis.fig_ablation import load_sweeps
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--patch', type=int, choices=MNIST_PATCH_SIZES, default=MNIST['patch'])
    parser.add_argument('--artifacts', type=Path, default=ARTIFACTS)
    parser.add_argument('--out-dir', type=Path, default=TABLES)
    args = parser.parse_args(argv)
    setup_logging()
    base = load_artifact(args.artifacts / 'base.npz', dict(cfg=BASE, seed=BASE_SEED))
    row = base_summary(base)
    log.info('base run: %s', row)
    write_csv(args.out_dir / 'base_summary.csv', [row])
    write_csv(args.out_dir / 'sweep_summary.csv', sweep_summary(base, load_sweeps(args.artifacts)))
    cfg = dict(MNIST, patch=args.patch)
    mnist = load_artifact(args.artifacts / f'mnist_p{args.patch}.npz', dict(cfg=cfg, seed=MNIST_SEED))
    row = mnist_summary(mnist)
    log.info('mnist run: %s', row)
    write_csv(args.out_dir / f'mnist_summary_p{args.patch}.csv', [row])


if __name__ == '__main__':
    cli(main)
