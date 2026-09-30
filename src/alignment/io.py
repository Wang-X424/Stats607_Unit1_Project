"""Saving and loading artifacts together with the configuration that produced them.

Every artifact stores its configuration as JSON. Loading with expected_config raises if the
stored configuration differs, so a changed setting can never silently reuse old results.
"""
import json
import logging
import sys

import numpy as np


def setup_logging(level=logging.INFO):
    """Timestamped progress messages on stderr."""
    logging.basicConfig(level=level, stream=sys.stderr,
                        format='%(asctime)s %(levelname)s %(name)s: %(message)s', datefmt='%H:%M:%S')


def _canonical(config):
    return json.dumps(config, sort_keys=True, default=list)


def save_artifact(path, config, **arrays):
    """Write arrays and their configuration to a compressed .npz file (parent dirs created)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, config=np.array(_canonical(config)), **arrays)
    logging.getLogger(__name__).info('wrote %s', path)


def load_artifact(path, expected_config=None):
    """Load an artifact as a dict of arrays (plus 'config' as a dict).

    Raises FileNotFoundError naming the command that creates it, and RuntimeError if
    expected_config is given and differs from the stored configuration.
    """
    if not path.exists():
        raise FileNotFoundError(f'{path} not found. Run `make pipeline` (full run) or restore the '
                                f'committed checkpoint with `git checkout -- artifacts/`.')
    with np.load(path) as f:
        out = {k: f[k] for k in f.files}
    stored = str(out.pop('config'))
    if expected_config is not None and stored != _canonical(expected_config):
        raise RuntimeError(f'{path} was produced with a different configuration than the one in '
                           f'alignment/config.py; rerun the pipeline (`make pipeline`).')
    out['config'] = json.loads(stored)
    return out


def cli(main):
    """Run a command-line main(); report expected failures as one clear line and exit with status 1."""
    try:
        main()
    except (FileNotFoundError, RuntimeError, ValueError) as err:
        logging.getLogger('alignment').error('%s', err)
        sys.exit(1)
