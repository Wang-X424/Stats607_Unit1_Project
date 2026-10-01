"""Train the base configuration with one seed and save the full training log.

Output: artifacts/runs/base.npz (training log, the 2-D signal patches, labels and u_star).
Usage: python -m pipeline.run_base [--out-dir DIR]
"""
import argparse
import logging
import time
from pathlib import Path

from alignment.config import ARTIFACTS, BASE, BASE_SEED
from alignment.io import cli, save_artifact, setup_logging
from alignment.model import run_one

log = logging.getLogger('pipeline.run_base')


def run_base(cfg, seed, out_path):
    """Run one seed of cfg and write the log plus the signal-plane data to out_path."""
    start = time.time()
    log.info('base run: %d iterations, seed %d', cfg['iters'], seed)
    X, y, u_star, trace, _ = run_one(cfg, seed)
    save_artifact(out_path, dict(cfg=cfg, seed=seed),
                  signals=X[:, 0, :2], y=y, u_star=u_star[:2], **trace)
    log.info('base run done in %.1f s', time.time() - start)


def main(argv=None):
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--out-dir', type=Path, default=ARTIFACTS)
    args = parser.parse_args(argv)
    setup_logging()
    run_base(BASE, BASE_SEED, args.out_dir / 'base.npz')


if __name__ == '__main__':
    cli(main)
