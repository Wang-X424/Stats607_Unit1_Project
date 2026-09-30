"""Shared pieces for the simulation studies: two-cone data model, gradient descent,
recorded quantities, the base run and the parameter sweeps (all cached in results/)."""
import os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS, FIGS = os.path.join(HERE, 'results'), os.path.join(HERE, 'figs')
os.makedirs(RESULTS, exist_ok=True)
os.makedirs(FIGS, exist_ok=True)

# Base configuration (Section 1.2 of the experiment plan). theta_deg is theta_star in degrees,
# split equally between the cone half-angle alpha and the tilt of the class axes (pi - beta_0)/2.
BASE = dict(D=100, P=5, n=40, h=50, theta_deg=24.0, snr=10.0, init_scale=1e-6,
            eta=2e-3, iters=40_000, log_every=20, rho=1.0, b0=0.25)

SWEEPS = {
    'eps':   dict(key='init_scale', values=[1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8],
                  xlabel=r'initialization scale $\epsilon$', xscale='log'),
    # 'fixed' overrides BASE for this sweep only: at SNR 10 even P = 17 keeps G an order of
    # magnitude below the class mass, so the P sweep is run at SNR 1 (G ~ 6 sqrt(P-1)).
    'P':     dict(key='P', values=[2, 3, 5, 9, 17], fixed=dict(snr=1.0),
                  xlabel=r'number of patches $P$', xscale='log', int_ticks=True),
    'G':     dict(key='snr', values=[100, 30, 10, 3, 1, 0.3],
                  xlabel=r'noise level $\hat G$', xscale='log'),
    'theta': dict(key='theta_deg', values=[5, 10, 15, 20, 24, 30, 35, 40, 44],
                  xlabel=r'$\theta_\star$ (degrees)', xscale='linear'),
}


# ----------------------------------------------------------------------------- data
def make_data(cfg, rng):
    """Two-cone signal-noise patch model. Returns X (n, P, D), labels y (n,), common axis u_star (D,).
    The signal plane is span{e_1, e_2}; noise patches live in the remaining D-2 coordinates."""
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
    """Lower estimate of G = sup_w ||g(w)||, g(w) = sum of y_i * (noise patches activated by w),
    by iterating w <- g(w)/||g(w)|| from random starts and from the coherent direction sum y_i xi."""
    Z = X[:, 1:, :].reshape(-1, X.shape[2])
    Y = np.repeat(y, X.shape[1] - 1)
    g = lambda w: ((Z @ w > 0) * Y) @ Z
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


# ------------------------------------------------------------------------- training
def train(X, y, u_star, cfg, rng, refs=None):
    """Full-batch gradient descent on the mean exponential loss, both layers trained, from the
    unbalanced Gaussian initialization W_ij, v_j ~ N(0, eps^2). Records every log_every steps:
    loss, max_i |f_i|, c_j = cos(w_j, sgn(v_j) u_star), sgn(v_j), Q_+/Q_-, fraction of filters
    in S_+/S_- and 'alive' flags (both read patch 0 as the signal patch: synthetic data only),
    the projection of w_j onto the first two coordinates, ||w_j||, the stable rank of W, and
    c_lam = cos(w_j, sgn(v_j) u_lam) with u_lam the loss-weighted signed patch sum (the
    direction currently pulling the filters). If refs (D, m) is given, also cos(w_j, refs)."""
    n, P, D = X.shape
    h, eta, eps = cfg['h'], cfg['eta'], cfg['init_scale']
    W, v = rng.normal(0, eps, (D, h)), rng.normal(0, eps, h)
    Xf, Xsum = X.reshape(n * P, D), X.sum(1)
    S, pos = X[:, 0, :], y > 0                                 # signal patches, positive inputs
    log = {k: [] for k in ['iter', 'loss', 'maxf', 'c', 'sgn', 'Qp', 'Qm', 'fracS', 'alive', 'proj',
                           'norm', 'srank', 'c_lam'] + (['cref'] if refs is not None else [])}

    for k in range(cfg['iters'] + 1):
        Z = Xf @ W                                             # pre-activations, (nP, h)
        Phi = np.maximum(Z, 0).reshape(n, P, h).sum(1)         # pooled responses, (n, h)
        f = Phi @ v
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

        g = -(y * lam) / n                                     # dL/df_i
        grad_W = Xf.T @ ((Z > 0) * (np.repeat(g, P)[:, None] * v[None, :]))
        v -= eta * (Phi.T @ g)
        W -= eta * grad_W
    return {k: np.array(val) for k, val in log.items()}


ROUND = 1000   # snapshot iterations shown in the figures are rounded to a multiple of this


def first_iter(log, cond, fallback):
    """First recorded iteration at which cond (boolean array over the log) holds."""
    idx = np.flatnonzero(cond)
    return int(log['iter'][idx[0]]) if len(idx) else fallback


def round_iter(k, log):
    """Round an iteration to the nearest multiple of ROUND (halves up), inside the recorded range."""
    return int(np.clip(np.floor(k / ROUND + 0.5) * ROUND, log['iter'][1], log['iter'][-1]))


def class_angle(log, cfg):
    """Angle in degrees between the signal-plane projection of each filter and the axis of its
    class (u_+ for v_j > 0, u_- for v_j < 0), for every recorded step; shape (logs, h)."""
    axis = np.where(log['sgn'] > 0, 0.0, 180.0 - cfg['theta_deg'])
    ang = np.degrees(np.arctan2(log['proj'][..., 1], log['proj'][..., 0]))
    return np.abs((ang - axis + 180) % 360 - 180)


def landmarks(log, cfg=BASE):
    """Measured alignment time and t_2. A filter that activates no signal patch at the end is
    'dead': its gradient lies in the noise subspace, so its signal-plane direction is frozen for
    good (the analogue of MMV's S_dead). t_1 is the first iteration at which every other
    filter points inside the cone of its class (angle <= alpha = theta_star/2); t_2 the first at
    which Q_+ and Q_- both exceed b0."""
    alive = log['alive'][-1]
    inside = class_angle(log, cfg)[:, alive] <= cfg['theta_deg'] / 2
    t_1 = first_iter(log, inside.all(1), int(log['iter'][-1]))
    t2 = first_iter(log, (log['Qp'] >= cfg['b0']) & (log['Qm'] >= cfg['b0']), int(log['iter'][-1]))
    return t_1, t2


def run_one(cfg, seed):
    rng = np.random.default_rng(seed)
    X, y, u_star = make_data(cfg, rng)
    return X, y, u_star, train(X, y, u_star, cfg, rng), rng


# ------------------------------------------------------------------ cached experiments
def base_run(seed=0):
    """The base configuration, one seed. Returns the training log plus the signal patches."""
    path = os.path.join(RESULTS, 'base.npz')
    if not os.path.exists(path):
        X, y, u_star, log, _ = run_one(BASE, seed)
        np.savez(path, signals=X[:, 0, :2], y=y, u_star=u_star[:2], **log)
    return dict(np.load(path))


def landmark_iters():
    """The four iterations at which the sweeps are read out: roughly t_1/2, t_1 and t_2
    of the base run (rounded to multiples of ROUND), and the last iteration."""
    log = base_run()
    t_1, t2 = landmarks(log)
    return [round_iter(t_1 / 2, log), round_iter(t_1, log), round_iter(t2, log), int(log['iter'][-1])]


def run_sweep(name, seeds=range(5)):
    """One parameter sweep, all seeds (cached). Returns values, iters, Ghat and the fraction of
    dead filters (values x seeds), and the full curves cbar = mean of c_j over the live filters
    and loss (values x seeds x logs)."""
    sw = SWEEPS[name]
    fixed = sw.get('fixed', {})
    path = os.path.join(RESULTS, f'sweep_{name}' + ''.join(f'_{k}{v}' for k, v in fixed.items()) + '.npz')
    if not os.path.exists(path):
        out = {'cbar': [], 'loss': [], 'Ghat': [], 'dead': []}
        for val in sw['values']:
            cfg = dict(BASE, **fixed, **{sw['key']: val})
            rows = {k: [] for k in out}
            for seed in seeds:
                X, y, _, log, rng = run_one(cfg, seed)
                alive = log['alive'][-1]
                rows['cbar'].append(log['c'][:, alive].mean(1) if alive.any() else np.zeros(len(log['iter'])))
                rows['loss'].append(log['loss'])
                rows['Ghat'].append(noise_level(X, y, rng))
                rows['dead'].append(1 - alive.mean())
            for k in out:
                out[k].append(rows[k])
            print(f'sweep {name}: {sw["key"]} = {val} done', flush=True)
        np.savez(path, values=np.array(sw['values'], float), iters=log['iter'],
                 **{k: np.array(val) for k, val in out.items()})
    return dict(np.load(path))


def plot_sweeps(metric, ylabel, yscale, fname, ylim=None, seeds=range(5)):
    """One figure per sweep: a row with one panel per landmark iteration (shared y);
    median over seeds with min-max band. Files are named <fname>_<sweep>."""
    import matplotlib.pyplot as plt
    ks = landmark_iters()
    for name, sw in SWEEPS.items():
        res = run_sweep(name, seeds)
        x = np.median(res['Ghat'], 1) if name == 'G' else res['values']
        fig, row = plt.subplots(1, len(ks), figsize=(3.2 * len(ks), 2.7), sharey=True)
        for ax, k in zip(row, ks):
            Y = res[metric][:, :, np.searchsorted(res['iters'], k)]      # values x seeds
            ax.plot(x, np.median(Y, 1), 'o-', ms=3)
            ax.fill_between(x, Y.min(1), Y.max(1), alpha=0.25)
            ax.set(xscale=sw['xscale'], yscale=yscale, xlabel=sw['xlabel'], title=f'iteration {k}')
            if sw.get('int_ticks'):
                ax.set_xticks(x, [str(int(v)) for v in x], minor=False)
                ax.set_xticks([], minor=True)
        row[0].set_ylabel(ylabel)
        if ylim:
            row[0].set_ylim(*ylim)
        fig.tight_layout()
        fig.savefig(os.path.join(FIGS, f'{fname}_{name}.pdf'))
        fig.savefig(os.path.join(FIGS, f'{fname}_{name}.png'), dpi=150)
        plt.close(fig)
