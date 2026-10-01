"""Parameter sweeps: vary one parameter of the base configuration over a grid, several seeds each.

Output: artifacts/runs/sweep_<name>.npz for name in eps, P, G, theta, holding
values, iters, Ghat and the fraction of dead filters (values x seeds), and the full curves
cbar = mean of c_j over the live filters and loss (values x seeds x logs).
Usage: python -m pipeline.run_sweeps [--only NAME ...] [--out-dir DIR]
"""
import argparse
import logging
import time
from pathlib import Path

import numpy as np

from alignment.config import ARTIFACTS, BASE, SWEEP_SEEDS, SWEEPS
from alignment.io import cli, save_artifact, setup_logging
from alignment.model import noise_level, run_one

log = logging.getLogger('pipeline.run_sweeps')


def sweep_config(name, base=BASE, sweeps=SWEEPS, seeds=SWEEP_SEEDS):
    """The configuration stored with (and expected from) the artifact of one sweep."""
    return dict(base=base, sweep=sweeps[name], seeds=list(seeds))


def run_sweep(name, out_path, base=BASE, sweeps=SWEEPS, seeds=SWEEP_SEEDS):
    """Run all values x seeds of one sweep and write the summary curves to out_path."""
    sw = sweeps[name]
    fixed = sw.get('fixed', {})
    out = {'cbar': [], 'loss': [], 'Ghat': [], 'dead': []}
    for i, val in enumerate(sw['values'], 1):
        start = time.time()
        cfg = dict(base, **fixed, **{sw['key']: val})
        rows = {k: [] for k in out}
        for seed in seeds:
            X, y, _, trace, rng = run_one(cfg, seed)
            alive = trace['alive'][-1]
            rows['cbar'].append(trace['c'][:, alive].mean(1) if alive.any()
                                else np.zeros(len(trace['iter'])))
            rows['loss'].append(trace['loss'])
            rows['Ghat'].append(noise_level(X, y, rng))
            rows['dead'].append(1 - alive.mean())
        for k in out:
            out[k].append(rows[k])
        log.info('sweep %s [%d/%d]: %s = %g done (%d seeds, %.1f s)', name, i, len(sw['values']),
                 sw['key'], val, len(seeds), time.time() - start)
    save_artifact(out_path, sweep_config(name, base, sweeps, seeds),
                  values=np.array(sw['values'], float), iters=trace['iter'],
                  **{k: np.array(v) for k, v in out.items()})


def main(argv=None):
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--only', nargs='+', choices=list(SWEEPS), default=list(SWEEPS),
                        help='run only these sweeps (default: all)')
    parser.add_argument('--out-dir', type=Path, default=ARTIFACTS)
    args = parser.parse_args(argv)
    setup_logging()
    for name in args.only:
        run_sweep(name, args.out_dir / f'sweep_{name}.npz')


if __name__ == '__main__':
    cli(main)
