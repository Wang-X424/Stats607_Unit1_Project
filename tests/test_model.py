"""Function correctness: gradients, landmark helpers and determinism of training."""
import numpy as np

from alignment.metrics import class_angle, first_iter, landmarks, round_iter
from alignment.model import loss_and_grad, make_data, run_one


def test_gradient_matches_finite_differences(small_cfg):
    """Analytic gradients of the mean exponential loss agree with central differences."""
    rng = np.random.default_rng(0)
    X, y, _ = make_data(small_cfg, rng)
    W, v = rng.normal(0, 0.5, (small_cfg['D'], small_cfg['h'])), rng.normal(0, 0.5, small_cfg['h'])
    _, gW, gv = loss_and_grad(W, v, X, y)
    h = 1e-6
    num_gW = np.zeros_like(W)
    for idx in np.ndindex(W.shape):
        E = np.zeros_like(W)
        E[idx] = h
        num_gW[idx] = (loss_and_grad(W + E, v, X, y)[0] - loss_and_grad(W - E, v, X, y)[0]) / (2 * h)
    num_gv = np.array([(loss_and_grad(W, v + h * e, X, y)[0] - loss_and_grad(W, v - h * e, X, y)[0]) / (2 * h)
                       for e in np.eye(len(v))])
    np.testing.assert_allclose(gW, num_gW, rtol=1e-4, atol=1e-8)
    np.testing.assert_allclose(gv, num_gv, rtol=1e-4, atol=1e-8)


def test_training_is_deterministic_and_decreases_loss(small_cfg):
    """Same seed gives an identical log; a different seed does not; the loss goes down."""
    log_a, log_b, log_c = (run_one(small_cfg, s)[3] for s in (0, 0, 1))
    assert all(np.array_equal(log_a[k], log_b[k]) for k in log_a)
    assert not np.array_equal(log_a['proj'], log_c['proj'])
    assert log_a['loss'][-1] < log_a['loss'][0]
    n_logs = small_cfg['iters'] // small_cfg['log_every'] + 1
    assert log_a['c'].shape == (n_logs, small_cfg['h'])
    assert np.all(np.abs(log_a['c']) <= 1 + 1e-12)


def test_class_angle_wraps_around():
    """Angles are measured the short way round the circle and never exceed 180 degrees."""
    deg = np.array([[350.0, 10.0, 180.0, -170.0]])
    log = dict(sgn=np.array([[1.0, 1.0, 1.0, -1.0]]),
               proj=np.stack([np.cos(np.radians(deg)), np.sin(np.radians(deg))], -1))
    # class axis 0 for v > 0, 180 - 24 = 156 for v < 0
    np.testing.assert_allclose(class_angle(log, dict(theta_deg=24.0)), [[10, 10, 180, 34]], atol=1e-9)


def test_iteration_helpers():
    """first_iter falls back when the event never happens; round_iter rounds halves up and clips."""
    log = dict(iter=np.arange(0, 10_001, 20))
    assert first_iter(log, log['iter'] >= 4_010, None) == 4_020
    assert first_iter(log, log['iter'] < 0, 'never') == 'never'
    assert round_iter(2_500, log) == 3_000 and round_iter(2_499, log) == 2_000
    assert round_iter(0, log) == 20 and round_iter(99_999, log) == 10_000


def test_landmarks_order(small_cfg):
    """t_1 and t_2 lie within the recorded range."""
    log = run_one(dict(small_cfg, iters=2_000), 0)[3]
    t_1, t_2 = landmarks(log, small_cfg)
    assert 0 <= t_1 <= log['iter'][-1] and 0 <= t_2 <= log['iter'][-1]
