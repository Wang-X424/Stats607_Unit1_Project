"""Data validation: the synthetic two-cone data and the MNIST patching have the promised structure."""
import numpy as np
import pytest

from alignment.config import BASE
from alignment.mnist import class_axes, load_mnist_01
from alignment.model import make_data


@pytest.mark.parametrize('theta_deg', [5.0, 24.0, 44.0])
def test_two_cone_geometry(theta_deg):
    """Shapes, balanced labels, signal/noise supports, and every signal patch inside its class cone."""
    cfg = dict(BASE, theta_deg=theta_deg)
    X, y, u_star = make_data(cfg, np.random.default_rng(0))
    n, P, D = cfg['n'], cfg['P'], cfg['D']
    assert X.shape == (n, P, D) and y.shape == (n,) and u_star.shape == (D,)
    assert np.isfinite(X).all()
    assert set(np.unique(y)) == {-1.0, 1.0} and y.sum() == 0
    # signal patch: in span{e1, e2} with norm rho; noise patches: zero on e1, e2
    assert np.allclose(X[:, 0, 2:], 0) and np.allclose(X[:, 1:, :2], 0)
    assert np.allclose(np.linalg.norm(X[:, 0], axis=1), cfg['rho'])
    # angle to the class axis (0 for y = +1, 180 - theta for y = -1) is at most alpha = theta/2
    ang = np.degrees(np.arctan2(X[:, 0, 1], X[:, 0, 0]))
    axis = np.where(y > 0, 0.0, 180.0 - theta_deg)
    dev = np.abs((ang - axis + 180) % 360 - 180)
    assert dev.max() <= theta_deg / 2 + 1e-9
    assert np.isclose(np.linalg.norm(u_star), 1.0) and np.allclose(u_star[2:], 0)


@pytest.mark.parametrize('patch', [4, 7, 14, 28])
def test_mnist_patching(patch):
    """Balanced labels, P = (28/patch)^2, D = patch^2, centred images, patches tile the image."""
    rng = np.random.default_rng(0)
    digits = {d: rng.integers(0, 256, (30, 28, 28), dtype=np.uint8) for d in (0, 1)}
    X, y = load_mnist_01(digits, dict(n_per_class=10, patch=patch), np.random.default_rng(1))
    k = 28 // patch
    assert X.shape == (20, k * k, patch * patch)
    assert (y[:10] == 1).all() and (y[10:] == -1).all()
    assert np.allclose(X.mean(0), 0)                            # centring
    # reassembling the patches gives back exactly the whole (centred) image, i.e. the patch=28 input
    whole, _ = load_mnist_01(digits, dict(n_per_class=10, patch=28), np.random.default_rng(1))
    rebuilt = X.reshape(20, k, k, patch, patch).transpose(0, 1, 3, 2, 4).reshape(20, 784)
    np.testing.assert_array_equal(rebuilt, whole[:, 0])


def test_mnist_rejects_bad_patch_size():
    """A patch size that does not divide 28 fails with a clear error."""
    digits = {d: np.zeros((5, 28, 28), np.uint8) for d in (0, 1)}
    with pytest.raises(ValueError, match='divide 28'):
        load_mnist_01(digits, dict(n_per_class=2, patch=5), np.random.default_rng(0))


def test_class_axes_centred_balanced_data_are_antipodal():
    """With balanced classes and centred inputs, the pooled class sums cancel: u_- = -u_+."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(20, 4, 9))
    X -= X.mean(0)
    y = np.repeat([1.0, -1.0], 10)
    u_plus, u_minus, u_star = class_axes(X, y)
    assert np.isclose(u_plus @ u_minus, -1.0) and np.allclose(u_star, u_plus)
