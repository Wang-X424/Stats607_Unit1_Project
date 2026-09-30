"""Two-cone data model, full-batch gradient descent and the quantities recorded during training.

Network: f(x) = sum_j v_j sum_p ReLU(<w_j, x^(p)>), both layers trained on the mean
exponential loss L = (1/n) sum_i exp(-y_i f(x_i)) from the unbalanced Gaussian
initialization W_ij, v_j ~ N(0, eps^2).
"""
import numpy as np


# ----------------------------------------------------------------------------- data
def make_data(cfg, rng):
    """Sample the two-cone signal-noise patch model.

    Patch 0 of every input is the signal patch: a point of norm rho in the plane span{e_1, e_2},
    at angle at most alpha = theta_star/2 from the axis of its class. Patches 1..P-1 are
    Gaussian noise supported on the remaining D-2 coordinates.

    Returns
    -------
    X : ndarray (n, P, D)
    y : ndarray (n,), first half +1, second half -1
    u_star : ndarray (D,), unit common axis (u_+ - u_-)/||u_+ - u_-||
    """
    D, P, n, rho = cfg['D'], cfg['P'], cfg['n'], cfg['rho']
    theta = np.deg2rad(cfg['theta_deg'])
    alpha, beta0 = theta / 2, np.pi - theta
    y = np.repeat([1.0, -1.0], n // 2)
    axis = np.where(y > 0, 0.0, beta0)                         # angle of the class axis u_{y_i}
    ang = axis + rng.uniform(0, alpha, n) * rng.choice([-1.0, 1.0], n)
    X = np.zeros((n, P, D))
    X[:, 0, 0], X[:, 0, 1] = rho * np.cos(ang), rho * np.sin(ang)
    X[:, 1:, 2:] = rng.normal(0, rho / (cfg['snr'] * np.sqrt(D)), (n, P - 1, D - 2))
    u_plus, u_minus = np.array([1.0, 0.0]), np.array([np.cos(beta0), np.sin(beta0)])
    u_star = np.zeros(D)
    u_star[:2] = (u_plus - u_minus) / np.linalg.norm(u_plus - u_minus)
    return X, y, u_star


def noise_level(X, y, rng, n_starts=20, n_iter=10):
    """Lower estimate of G = sup_w ||g(w)||, with g(w) the sum of y_i times the noise patches
    activated by w.

    Iterates w <- g(w)/||g(w)|| from random starts and from the coherent direction sum y_i xi.
    """
    Z = X[:, 1:, :].reshape(-1, X.shape[2])
    Y = np.repeat(y, X.shape[1] - 1)
    g = lambda w: ((Z @ w > 0) * Y) @ Z                        # noqa: E731
    starts = list(rng.normal(size=(n_starts, Z.shape[1]))) + [Y @ Z, -(Y @ Z)]
    best = 0.0
    for w in starts:
        for _ in range(n_iter):
            gw = g(w / np.linalg.norm(w))
            size = np.linalg.norm(gw)
            best = max(best, size)
            if size == 0:
                break
            w = gw
    return best


# ------------------------------------------------------------------------- forward / backward
def forward(W, v, Xf, n, P):
    """Pre-activations Z (nP, h), pooled responses Phi (n, h) and outputs f (n,)."""
    Z = Xf @ W
    Phi = np.maximum(Z, 0).reshape(n, P, W.shape[1]).sum(1)
    return Z, Phi, Phi @ v


def backward(Xf, Z, Phi, v, y, lam, P):
    """Gradients of the mean exponential loss with respect to W (D, h) and v (h,).

    lam are the per-sample losses exp(-y_i f_i), so dL/df_i = -y_i lam_i / n.
    """
    g = -(y * lam) / len(y)
    grad_W = Xf.T @ ((Z > 0) * (np.repeat(g, P)[:, None] * v[None, :]))
    return grad_W, Phi.T @ g


def loss_and_grad(W, v, X, y):
    """Mean exponential loss and its gradients (grad_W, grad_v) at (W, v) for inputs X (n, P, D)."""
    n, P, D = X.shape
    Xf = X.reshape(n * P, D)
    Z, Phi, f = forward(W, v, Xf, n, P)
    lam = np.exp(np.clip(-y * f, -50, 50))
    grad_W, grad_v = backward(Xf, Z, Phi, v, y, lam, P)
    return lam.mean(), grad_W, grad_v


# ------------------------------------------------------------------------- training
LOG_KEYS = ['iter', 'loss', 'maxf', 'c', 'sgn', 'Qp', 'Qm', 'fracS', 'alive', 'proj',
            'norm', 'srank', 'c_lam']


def train(X, y, u_star, cfg, rng, refs=None):
    """Full-batch gradient descent; returns the training log as a dict of arrays.

    Recorded every cfg['log_every'] steps:
    loss, max_i |f_i|, c_j = cos(w_j, sgn(v_j) u_star), sgn(v_j), Q_+ / Q_-, the fraction of
    filters in S_+ / S_-, 'alive' flags (at least one signal patch activated; both read patch 0
    as the signal patch, so they are meaningful for synthetic data only), the projection of w_j
    onto the first two coordinates, ||w_j||, the stable rank of W, and
    c_lam = cos(w_j, sgn(v_j) u_lam) with u_lam the loss-weighted signed patch sum.
    If refs (D, m) is given, also cref = cos(w_j, refs), shape (logs, h, m).
    """
    n, P, D = X.shape
    h, eta, eps = cfg['h'], cfg['eta'], cfg['init_scale']
    W, v = rng.normal(0, eps, (D, h)), rng.normal(0, eps, h)
    Xf, Xsum = X.reshape(n * P, D), X.sum(1)
    S, pos = X[:, 0, :], y > 0                                 # signal patches, positive inputs
    log = {k: [] for k in LOG_KEYS + (['cref'] if refs is not None else [])}

    for k in range(cfg['iters'] + 1):
        Z, Phi, f = forward(W, v, Xf, n, P)
        lam = np.exp(np.clip(-y * f, -50, 50))                 # loss weights = per-sample losses

        if k % cfg['log_every'] == 0:
            norm = np.linalg.norm(W, axis=0)
            sgn = np.where(v >= 0, 1.0, -1.0)
            a = u_star @ W
            A = S @ W > 0                                      # activated signal patches, (n, h)
            in_S = np.where(sgn > 0, A[pos].all(0) & ~A[~pos].any(0),
                                     A[~pos].all(0) & ~A[pos].any(0))
            u_lam = (y * lam) @ Xsum
            u_lam /= np.linalg.norm(u_lam)
            for key, val in [('iter', k), ('loss', lam.mean()), ('maxf', np.abs(f).max()),
                             ('c', a / norm * sgn), ('sgn', sgn), ('Qp', (a[sgn > 0] ** 2).sum()),
                             ('Qm', (a[sgn < 0] ** 2).sum()), ('fracS', in_S.mean()),
                             ('alive', A.any(0)), ('proj', W[:2].T.copy()), ('norm', norm),
                             ('srank', (norm ** 2).sum() / np.linalg.norm(W, 2) ** 2),
                             ('c_lam', (u_lam @ W) / norm * sgn)]:
                log[key].append(val)
            if refs is not None:
                log['cref'].append((W / norm).T @ refs)

        grad_W, grad_v = backward(Xf, Z, Phi, v, y, lam, P)
        v -= eta * grad_v
        W -= eta * grad_W
    return {k: np.array(val) for k, val in log.items()}


def run_one(cfg, seed):
    """Sample data and train with one seed; returns X, y, u_star, log and the generator.

    The generator is returned in its post-training state because noise_level draws from it.
    """
    rng = np.random.default_rng(seed)
    X, y, u_star = make_data(cfg, rng)
    return X, y, u_star, train(X, y, u_star, cfg, rng), rng
