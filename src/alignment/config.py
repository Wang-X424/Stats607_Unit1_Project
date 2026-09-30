"""Experiment configurations and repository paths.

Every number that affects a result lives here, so that no script has to be edited by hand
to regenerate the figures. Paths are resolved relative to the repository root.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = ROOT / 'data' / 'raw'
DATA_PROCESSED = ROOT / 'data' / 'processed'
ARTIFACTS = ROOT / 'artifacts' / 'runs'
FIGURES = ROOT / 'results' / 'figures'
TABLES = ROOT / 'results' / 'tables'

# Base configuration. theta_deg is theta_star in degrees, split equally between the cone
# half-angle alpha and the tilt of the class axes (pi - beta_0)/2.
BASE = dict(D=100, P=5, n=40, h=50, theta_deg=24.0, snr=10.0, init_scale=1e-6,
            eta=2e-3, iters=40_000, log_every=20, rho=1.0, b0=0.25)
BASE_SEED = 0

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
SWEEP_SEEDS = list(range(5))

# MNIST digits 0 vs 1: 14x14 patches -> P = 4, D = 196.
MNIST = dict(BASE, h=50, init_scale=1e-6, eta=2e-3, iters=15_000, log_every=20,
             n_per_class=1000, patch=14)
MNIST_SEED = 0
MNIST_PATCH_SIZES = (4, 7, 14, 28)

ROUND = 1000   # snapshot iterations shown in the figures are rounded to a multiple of this
