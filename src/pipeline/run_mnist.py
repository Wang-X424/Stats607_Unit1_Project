"""Train the network on MNIST 0 vs 1 and save the training log.

The same generator (seed MNIST_SEED) first draws the balanced subsample and then the
initialization, exactly as in the original analysis.
Input:  data/processed/mnist01.npz (from pipeline.prepare_mnist)
Output: artifacts/runs/mnist_p<patch>.npz
Usage: python -m pipeline.run_mnist [--patch {4,7,14,28}] [--out-dir DIR]
"""
import argparse
import logging
import time
from pathlib import Path

import numpy as np

from alignment.config import ARTIFACTS, DATA_PROCESSED, MNIST, MNIST_PATCH_SIZES, MNIST_SEED
from alignment.io import cli, save_artifact, setup_logging
from alignment.mnist import class_axes, geometry_report, load_mnist_01
from alignment.model import train

log = logging.getLogger('pipeline.run_mnist')


def run_mnist(cfg, seed, processed_path, out_path):
    """Subsample, patch, train and write the log plus the class axes to out_path."""
    if not processed_path.exists():
        raise FileNotFoundError(f'{processed_path} not found; run `make data` '
                                f'(python -m pipeline.prepare_mnist) first')
    with np.load(processed_path) as f:
        digits = {1: f['ones'], 0: f['zeros']}
    rng = np.random.default_rng(seed)
    X, y = load_mnist_01(digits, cfg, rng)
    u_plus, u_minus, u_star = class_axes(X, y)
    geo = geometry_report(X, y, u_star)
    log.info('n = %d, P = %d, D = %d; signed patches kept %.2f, cos with u_star: median %.2f, '
             '10%% quantile %.2f, fraction > 0 %.2f', len(y), X.shape[1], X.shape[2],
             geo['kept'], geo['cos_median'], geo['cos_q10'], geo['cos_positive'])
    start = time.time()
    trace = train(X, y, u_star, cfg, rng, refs=np.stack([u_plus, u_minus], 1))
    log.info('training done in %.1f s', time.time() - start)
    save_artifact(out_path, dict(cfg=cfg, seed=seed), u_plus=u_plus, u_minus=u_minus,
                  shape=np.array(X.shape), **{f'geo_{k}': v for k, v in geo.items()}, **trace)


def main(argv=None):
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--patch', type=int, choices=MNIST_PATCH_SIZES, default=MNIST['patch'])
    parser.add_argument('--processed', type=Path, default=DATA_PROCESSED / 'mnist01.npz')
    parser.add_argument('--out-dir', type=Path, default=ARTIFACTS)
    args = parser.parse_args(argv)
    setup_logging()
    cfg = dict(MNIST, patch=args.patch)
    run_mnist(cfg, MNIST_SEED, args.processed, args.out_dir / f'mnist_p{args.patch}.npz')


if __name__ == '__main__':
    cli(main)
