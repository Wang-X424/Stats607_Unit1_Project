"""Pipeline integrity: every stage runs end to end on a small configuration and writes its outputs;
stale or missing artifacts fail loudly; the full base run reproduces the original results."""
import numpy as np
import pytest

from alignment.config import BASE, BASE_SEED, SWEEPS
from alignment.io import load_artifact, save_artifact
from alignment.model import LOG_KEYS
from analysis import fig_ablation, fig_base, fig_mnist, make_tables
from pipeline.run_base import run_base
from pipeline.run_mnist import run_mnist
from pipeline.run_sweeps import run_sweep


def test_simulation_pipeline_end_to_end(tmp_path, small_cfg):
    """base run + two sweeps -> artifacts -> all base / ablation figures and the tables."""
    art, figs = tmp_path / 'artifacts', tmp_path / 'figures'
    run_base(small_cfg, 0, art / 'base.npz')
    sweeps = {k: dict(SWEEPS[k], values=SWEEPS[k]['values'][:2]) for k in ('eps', 'P')}
    for name in sweeps:
        run_sweep(name, art / f'sweep_{name}.npz', base=small_cfg, sweeps=sweeps, seeds=[0, 1])
    base = load_artifact(art / 'base.npz', dict(cfg=small_cfg, seed=0))
    assert set(LOG_KEYS) <= set(base)
    loaded = {n: load_artifact(art / f'sweep_{n}.npz') for n in sweeps}
    assert loaded['P']['cbar'].shape[:2] == (2, 2)

    fig_base.make_figures(base, figs)
    fig_ablation.make_figures(base, loaded, figs)
    expected = [f'base_experiment_{t}' for t in ('normalized', 'raw', 'loss')] + \
               [f'ablation_{m}_{n}' for m in ('alignment', 'loss') for n in sweeps]
    for stem in expected:
        for ext in ('pdf', 'png'):
            assert (figs / f'{stem}.{ext}').stat().st_size > 0, stem

    row = make_tables.base_summary(base)
    assert row['filters'] == small_cfg['h'] and 0 <= row['dead_filters'] <= row['filters']
    rows = make_tables.sweep_summary(base, loaded)
    assert len(rows) == 4 and all(-1 <= r[k] <= 1 for r in rows for k in r if k.startswith('cbar'))


def test_mnist_pipeline_end_to_end(tmp_path):
    """processed digits -> MNIST run -> figures and summary, on random stand-in images."""
    rng = np.random.default_rng(0)
    processed = tmp_path / 'mnist01.npz'
    np.savez(processed, ones=rng.integers(0, 256, (40, 28, 28), dtype=np.uint8),
             zeros=rng.integers(0, 256, (40, 28, 28), dtype=np.uint8))
    cfg = dict(BASE, h=4, iters=200, log_every=20, n_per_class=10, patch=14)
    run_mnist(cfg, 0, processed, tmp_path / 'mnist_p14.npz')
    run = load_artifact(tmp_path / 'mnist_p14.npz', dict(cfg=cfg, seed=0))
    assert run['cref'].shape == (11, 4, 2) and list(run['shape']) == [20, 4, 196]
    fig_mnist.make_figures(run, tmp_path / 'figures')
    assert (tmp_path / 'figures' / 'mnist_experiment_p14.png').exists()
    assert make_tables.mnist_summary(run)['n'] == 20


def test_missing_inputs_fail_clearly(tmp_path):
    """Missing artifacts and missing processed data name the command that creates them."""
    with pytest.raises(FileNotFoundError, match='make pipeline'):
        load_artifact(tmp_path / 'nope.npz')
    with pytest.raises(FileNotFoundError, match='make data'):
        run_mnist(dict(BASE, n_per_class=2, patch=14), 0, tmp_path / 'nope.npz', tmp_path / 'out.npz')


def test_stale_artifact_is_rejected(tmp_path, small_cfg):
    """An artifact made with a different configuration is not silently reused."""
    save_artifact(tmp_path / 'base.npz', dict(cfg=small_cfg, seed=0), x=np.zeros(1))
    load_artifact(tmp_path / 'base.npz', dict(cfg=small_cfg, seed=0))          # matches: fine
    with pytest.raises(RuntimeError, match='different configuration'):
        load_artifact(tmp_path / 'base.npz', dict(cfg=dict(small_cfg, eta=1e-3), seed=0))


@pytest.mark.slow
def test_base_run_reproduces_original_results(tmp_path):
    """The full base configuration gives the numbers recorded with the original scripts."""
    run_base(BASE, BASE_SEED, tmp_path / 'base.npz')
    row = make_tables.base_summary(load_artifact(tmp_path / 'base.npz'))
    assert row == dict(t_1=5780, t_S=2400, t_2=12180, t_loss_005=17400,
                       snapshots='0 3000 6000 12000 17000', final_loss=0.00445,
                       dead_filters=10, filters=50, max_final_angle_deg=0.9)
