"""Real data: MNIST digits 0 vs 1 (Section 5, main text). Images are centred as in MMV and cut
into non-overlapping patches; no patch is designated as signal or noise. The class axes u_+ and
u_- are the normalized pooled sums of the positive / negative patches, and u_star their
normalized difference. Trains the same network with the same GD as the simulations and plots,
per second-layer sign, the alignment of the filters with u_+ / u_- and the training loss.
Usage: python mnist_experiment.py [patch_size]   (patch_size in {4, 7, 14, 28}; default 14)"""
import gzip
import os
import struct
import sys
import urllib.request
import numpy as np
import matplotlib.pyplot as plt
from common import BASE, HERE, FIGS, train, first_iter

DATA = os.path.join(HERE, 'data')
MIRRORS = ['https://ossci-datasets.s3.amazonaws.com/mnist/', 'https://storage.googleapis.com/cvdf-datasets/mnist/']
FILES = {'images': 'train-images-idx3-ubyte.gz', 'labels': 'train-labels-idx1-ubyte.gz'}

CFG = dict(BASE, h=50, init_scale=1e-6, eta=2e-3, iters=15_000, log_every=20,
           n_per_class=1000, patch=14)          # 14x14 patches -> P = 4, D = 196
COLORS = {'pos': 'tab:green', 'neg': 'tab:red'}


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


def class_axes(X, y):
    """u_+, u_-: normalized pooled sums of the positive / negative patches; u_star = (u_+ - u_-)/||.||."""
    u = np.stack([X[y > 0].sum((0, 1)), X[y < 0].sum((0, 1))])
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    return u[0], u[1], (u[0] - u[1]) / np.linalg.norm(u[0] - u[1])


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
    if len(sys.argv) > 1:
        CFG['patch'] = int(sys.argv[1])
    rng = np.random.default_rng(0)
    X, y = load_mnist_01(CFG, rng)
    u_plus, u_minus, u_star = class_axes(X, y)
    geometry_report(X, y, u_star)
    log = train(X, y, u_star, CFG, rng, refs=np.stack([u_plus, u_minus], 1))
    it, sgn, cref = log['iter'], log['sgn'], log['cref']        # cref: (logs, h, 2) = cos(w_j, u_+), cos(w_j, u_-)
    tag = f"p{CFG['patch']}"

    # main figure: alignment per second-layer sign (mean and range over the filters of that sign), loss
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    groups = [('pos', sgn > 0, cref[..., 0], r'(a) filters with $v_j>0$: $\cos(w_j,u_+)$'),
              ('neg', sgn < 0, cref[..., 1], r'(b) filters with $v_j<0$: $\cos(w_j,u_-)$')]
    for ax, (key, mask, cos, title) in zip(axes, groups):
        vals = np.where(mask, cos, np.nan)
        ax.plot(it, np.nanmean(vals, 1), color=COLORS[key])
        ax.fill_between(it, np.nanmin(vals, 1), np.nanmax(vals, 1), color=COLORS[key], alpha=0.2)
        ax.set(xlabel='iteration', ylabel='alignment', ylim=(-1.05, 1.05), title=title)
    axes[2].plot(it, log['loss'], color='tab:blue')
    axes[2].set(xlabel='iteration', ylabel='training loss', title='(c) loss curve', ylim=(0, 1.1))
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, f'mnist_experiment_{tag}.pdf'))
    fig.savefig(os.path.join(FIGS, f'mnist_experiment_{tag}.png'), dpi=150)

    # diagnostics (not for the paper): alignment with the fixed axis vs. with the moving,
    # loss-weighted axis u_lam(t), and the stable rank of W
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.6))
    ax1.plot(it, log['c'].mean(1), label=r'fixed axis $u_\star$')
    ax1.plot(it, log['c_lam'].mean(1), label=r'loss-weighted axis $u_\lambda(t)$')
    ax1.set(xlabel='iteration', ylabel='mean alignment', ylim=(-1.05, 1.05))
    ax1.legend(fontsize=8)
    ax2.plot(it, log['srank'])
    ax2.set(xlabel='iteration', ylabel=r'stable rank $\|W\|_F^2/\|W\|_2^2$')
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, f'mnist_diagnostics_{tag}.png'), dpi=150)

    c, peak = log['c'].mean(1), int(np.argmax(log['c'].mean(1)))
    print(f'n = {len(y)}, P = {X.shape[1]}, D = {X.shape[2]}, cos(u_+, u_-) = {u_plus @ u_minus:.3f}; '
          f'mean alignment with u_star peaks at {c[peak]:.3f} (iteration {it[peak]}, loss {log["loss"][peak]:.3f}), '
          f'ends at {c[-1]:.3f}; with the loss-weighted axis it ends at {log["c_lam"][-1].mean():.3f}; '
          f'stable rank {log["srank"][peak]:.2f} at the peak, {log["srank"][-1]:.2f} at the end; '
          f'loss halves at iteration {first_iter(log, log["loss"] < 0.5 * log["loss"][0], "never")}; '
          f'final loss {log["loss"][-1]:.3g}')


if __name__ == '__main__':
    main()
