"""Download the MNIST training files and extract the images of digits 1 and 0.

Input:  data/raw/train-{images-idx3,labels-idx1}-ubyte.gz (downloaded if missing, MD5-checked)
Output: data/processed/mnist01.npz with uint8 arrays 'ones' and 'zeros' (original order)
Usage: python -m pipeline.prepare_mnist [--raw-dir DIR] [--out PATH]
"""
import argparse
import logging
from pathlib import Path

import numpy as np

from alignment.config import DATA_PROCESSED, DATA_RAW
from alignment.io import cli, setup_logging
from alignment.mnist import FILES, download, extract_digits

log = logging.getLogger('pipeline.prepare_mnist')


def prepare(raw_dir, out_path):
    """Fetch and verify the raw files, then write the digit-1 and digit-0 image stacks."""
    paths = {k: download(name, raw_dir) for k, name in FILES.items()}
    digits = extract_digits(paths['images'], paths['labels'])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, ones=digits[1], zeros=digits[0])
    log.info('wrote %s (%d ones, %d zeros)', out_path, len(digits[1]), len(digits[0]))


def main(argv=None):
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--raw-dir', type=Path, default=DATA_RAW)
    parser.add_argument('--out', type=Path, default=DATA_PROCESSED / 'mnist01.npz')
    args = parser.parse_args(argv)
    setup_logging()
    prepare(args.raw_dir, args.out)


if __name__ == '__main__':
    cli(main)
