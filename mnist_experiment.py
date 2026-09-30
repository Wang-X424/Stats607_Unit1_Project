"""Real data: MNIST digits 0 vs 1 (Section 5, main text). Images are centred, cut into
non-overlapping patches, and the common axis u_star is the normalized difference of the two pooled
class axes. Trains the same network with the same GD as the simulations and plots the mean
alignment with u_star and the training loss."""
import gzip
import os
import struct
import urllib.request
import numpy as np
import matplotlib.pyplot as plt
from common import BASE, HERE, FIGS, train, first_iter

DATA = os.path.join(HERE, 'data')
MIRRORS = ['https://ossci-datasets.s3.amazonaws.com/mnist/', 'https://storage.googleapis.com/cvdf-datasets/mnist/']
FILES = {'images': 'train-images-idx3-ubyte.gz', 'labels': 'train-labels-idx1-ubyte.gz'}

CFG = dict(BASE, h=50, init_scale=1e-6, eta=2e-3, iters=15_000, log_every=20,
           n_per_class=1000, patch=14)          # 14x14 patches -> P = 4, D = 196


def download(name):
    """Fetch one MNIST file into data/ (skipped if present; the torchvision .gz files work too)."""
    os.makedirs(DATA, exist_ok=True)
    path = os.path.join(DATA, name)
    if not os.path.exists(path):
        for base in MIRRORS:
            try:
                urllib.request.urlretrieve(base + name, path)
                break
            except Exception:
                continue
        else:
            raise RuntimeError(f'could not download {name}; place it in {DATA} manually')
    return path


def read_idx(path):
    """Parse an IDX file (big-endian header: magic, then one uint32 per dimension)."""
    with gzip.open(path, 'rb') as f:
        data = f.read()
    ndim = data[3]
    shape = struct.unpack('>' + 'I' * ndim, data[4:4 + 4 * ndim])
    return np.frombuffer(data, dtype=np.uint8, offset=4 + 4 * ndim).reshape(shape)


def load_mnist_01(cfg, rng):
    """Balanced subsample of digits 1 (y = +1) and 0 (y = -1), centred, cut into patches.
    Returns X (n, P, D), y (n,). Centring the images equals centring every patch position."""
    imgs = read_idx(download(FILES['images'])).astype(float) / 255
    labels = read_idx(download(FILES['labels']))
    idx = np.concatenate([rng.choice(np.flatnonzero(labels == d), cfg['n_per_class'], replace=False) for d in (1, 0)])
    y = np.repeat([1.0, -1.0], cfg['n_per_class'])
    imgs = imgs[idx] - imgs[idx].mean(0)
    k, p = 28 // cfg['patch'], cfg['patch']
    X = imgs.reshape(-1, k, p, k, p).transpose(0, 1, 3, 2, 4).reshape(len(idx), k * k, p * p)
    return X, y


def common_axis(X, y):
    """u_star = (u_+ - u_-)/||.||, with u_+/- the normalized pooled class sums of the patches."""
    u = np.stack([X[y > 0].sum((0, 1)), X[y < 0].sum((0, 1))])
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    return (u[0] - u[1]) / np.linalg.norm(u[0] - u[1])


def geometry_report(X, y, u_star):
    """How far the real data are from the two-cone model: cosines between the signed patches
    y_i x_i^(p) and u_star (near-empty background patches excluded)."""
    g = (y[:, None, None] * X).reshape(-1, X.shape[2])
    norms = np.linalg.norm(g, axis=1)
    keep = norms > 0.1 * norms.max()
    cos = g[keep] @ u_star / norms[keep]
    print(f'signed patches: {keep.mean():.2f} kept ({1 - keep.mean():.2f} near-empty); '
          f'cos with u_star: median {np.median(cos):.2f}, 10% quantile {np.quantile(cos, 0.1):.2f}, '
          f'fraction > 0: {(cos > 0).mean():.2f}')


def main():
    rng = np.random.default_rng(0)
    X, y = load_mnist_01(CFG, rng)
    u_star = common_axis(X, y)
    geometry_report(X, y, u_star)
    log = train(X, y, u_star, CFG, rng)
    it, c = log['iter'], log['c']                       # c: (logs, h), cos(w_j, sgn(v_j) u_star)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.6))
    ax1.plot(it, c.mean(1), color='tab:blue')
    ax1.fill_between(it, c.min(1), c.max(1), alpha=0.25)
    ax1.set(xlabel='iteration', ylabel=r'$\cos(w_j,\ \mathrm{sgn}(v_j)\,u_\star)$', ylim=(-1.05, 1.05),
            title='(a) alignment: mean and range over filters')
    ax2.plot(it, log['loss'], color='tab:blue')
    ax2.set(xlabel='iteration', ylabel='training loss', title='(b) loss curve', ylim=(0, 1.1))
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, 'mnist_experiment.pdf'))
    fig.savefig(os.path.join(FIGS, 'mnist_experiment.png'), dpi=150)

    drop = first_iter(log, log['loss'] < 0.5 * log['loss'][0], 'never')
    print(f'n = {len(y)}, P = {X.shape[1]}, D = {X.shape[2]}; mean alignment at the end {c[-1].mean():.3f} '
          f'(min {c[-1].min():.3f}, max {c[-1].max():.3f}), peak of the mean {c.mean(1).max():.3f} at iteration '
          f'{it[np.argmax(c.mean(1))]}; loss halves at iteration {drop}; final loss {log["loss"][-1]:.3g}')


if __name__ == '__main__':
    main()
