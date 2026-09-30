"""MNIST digits 0 vs 1: download, decoding and patching.

Raw files (data/raw/) are the original gzipped IDX training files. The processed file
(data/processed/mnist01.npz) holds the decoded images of the digits 1 and 0, in their original
order, as uint8. Subsampling, centring and patching happen at training time from a seeded
generator, so the processed file does not depend on any experiment setting.
"""
import gzip
import hashlib
import logging
import struct
import urllib.request

import numpy as np

log = logging.getLogger(__name__)

MIRRORS = ['https://ossci-datasets.s3.amazonaws.com/mnist/',
           'https://storage.googleapis.com/cvdf-datasets/mnist/']
FILES = {'images': 'train-images-idx3-ubyte.gz', 'labels': 'train-labels-idx1-ubyte.gz'}
MD5 = {'train-images-idx3-ubyte.gz': 'f68b3c2dcbeaaa9fbdd348bbdeb94873',
       'train-labels-idx1-ubyte.gz': 'd53e105ee54ea40749a09fcbcd1e9432'}


def md5sum(path):
    """MD5 hex digest of a file."""
    return hashlib.md5(path.read_bytes()).hexdigest()


def download(name, raw_dir, verify=True):
    """Fetch one MNIST file into raw_dir unless already present, and check its checksum.

    Raises RuntimeError with manual instructions if no mirror is reachable or the file is corrupt.
    """
    raw_dir.mkdir(parents=True, exist_ok=True)
    path = raw_dir / name
    if not path.exists():
        for base in MIRRORS:
            try:
                log.info('downloading %s from %s', name, base)
                urllib.request.urlretrieve(base + name, path)
                break
            except Exception as err:                        # try the next mirror
                log.warning('mirror %s failed: %s', base, err)
                path.unlink(missing_ok=True)
        else:
            raise RuntimeError(f'could not download {name} from any mirror; download it manually '
                               f'(e.g. from {MIRRORS[0]}{name}) and place it in {raw_dir}')
    if verify and name in MD5 and md5sum(path) != MD5[name]:
        raise RuntimeError(f'{path} has an unexpected MD5 checksum (corrupt or partial download); '
                           f'delete it and run `make data` again')
    return path


def read_idx(path):
    """Parse a gzipped IDX file (big-endian header: magic, then one uint32 per dimension)."""
    with gzip.open(path, 'rb') as f:
        data = f.read()
    ndim = data[3]
    shape = struct.unpack('>' + 'I' * ndim, data[4:4 + 4 * ndim])
    return np.frombuffer(data, dtype=np.uint8, offset=4 + 4 * ndim).reshape(shape)


def extract_digits(images_path, labels_path, digits=(1, 0)):
    """Images (uint8, original order) of each requested digit, as a dict {digit: (m, 28, 28)}."""
    imgs, labels = read_idx(images_path), read_idx(labels_path)
    if imgs.shape[1:] != (28, 28) or len(imgs) != len(labels):
        raise ValueError(f'unexpected MNIST shapes: images {imgs.shape}, labels {labels.shape}')
    return {d: imgs[labels == d] for d in digits}


def load_mnist_01(digit_images, cfg, rng):
    """Balanced subsample of digits 1 (y = +1) and 0 (y = -1), centred, cut into patches.

    digit_images maps 1 and 0 to uint8 image stacks. Returns X (n, P, D), y (n,) with
    P = (28/patch)^2 and D = patch^2. Centring the images equals centring every patch position.
    """
    p = cfg['patch']
    if 28 % p:
        raise ValueError(f'patch size must divide 28, got {p}')
    m = cfg['n_per_class']
    sel = [digit_images[d][rng.choice(len(digit_images[d]), m, replace=False)] for d in (1, 0)]
    imgs = np.concatenate(sel).astype(float) / 255
    y = np.repeat([1.0, -1.0], m)
    imgs = imgs - imgs.mean(0)
    k = 28 // p
    X = imgs.reshape(-1, k, p, k, p).transpose(0, 1, 3, 2, 4).reshape(len(imgs), k * k, p * p)
    return X, y


def class_axes(X, y):
    """u_+, u_-: normalized pooled sums of the positive / negative patches; u_star = (u_+ - u_-)/||.||."""
    u = np.stack([X[y > 0].sum((0, 1)), X[y < 0].sum((0, 1))])
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    return u[0], u[1], (u[0] - u[1]) / np.linalg.norm(u[0] - u[1])


def geometry_report(X, y, u_star):
    """How far the real data are from the two-cone model.

    Cosines between the signed patches y_i x_i^(p) and u_star, excluding near-empty patches
    (norm below 10% of the largest). Returns a dict of summary statistics.
    """
    g = (y[:, None, None] * X).reshape(-1, X.shape[2])
    norms = np.linalg.norm(g, axis=1)
    keep = norms > 0.1 * norms.max()
    cos = g[keep] @ u_star / norms[keep]
    return dict(kept=keep.mean(), cos_median=np.median(cos), cos_q10=np.quantile(cos, 0.1),
                cos_positive=(cos > 0).mean())
