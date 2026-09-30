"""Landmark times and angles computed from a training log (see alignment.model.train)."""
import numpy as np

from .config import ROUND


def first_iter(log, cond, fallback):
    """First recorded iteration at which cond (boolean array over the log) holds, else fallback."""
    idx = np.flatnonzero(cond)
    return int(log['iter'][idx[0]]) if len(idx) else fallback


def round_iter(k, log, step=ROUND):
    """Round iteration k to the nearest multiple of step (halves up), clipped to the recorded range."""
    return int(np.clip(np.floor(k / step + 0.5) * step, log['iter'][1], log['iter'][-1]))


def class_angle(log, cfg):
    """Angle in degrees between each filter's signal-plane projection and the axis of its class.

    The class axis is u_+ (angle 0) for v_j > 0 and u_- (angle 180 - theta_star) for v_j < 0.
    Returns an array of shape (logs, h) with values in [0, 180].
    """
    axis = np.where(log['sgn'] > 0, 0.0, 180.0 - cfg['theta_deg'])
    ang = np.degrees(np.arctan2(log['proj'][..., 1], log['proj'][..., 0]))
    return np.abs((ang - axis + 180) % 360 - 180)


def landmarks(log, cfg):
    """Measured alignment time t_1 and fitting time t_2.

    A filter that activates no signal patch at the end is 'dead': its gradient lies in the noise
    subspace, so its signal-plane direction is frozen for good (the analogue of MMV's S_dead).
    t_1 is the first iteration at which every other filter points inside the cone of its class
    (angle <= alpha = theta_star/2); t_2 the first at which Q_+ and Q_- both exceed b0.
    Both fall back to the last iteration if the event never happens.
    """
    last = int(log['iter'][-1])
    alive = log['alive'][-1]
    inside = class_angle(log, cfg)[:, alive] <= cfg['theta_deg'] / 2
    t_1 = first_iter(log, inside.all(1), last)
    t_2 = first_iter(log, (log['Qp'] >= cfg['b0']) & (log['Qm'] >= cfg['b0']), last)
    return t_1, t_2


def landmark_iters(log, cfg):
    """The four iterations at which the sweeps are read out.

    Roughly t_1/2, t_1 and t_2 of the base run (rounded to multiples of ROUND), and the last one.
    """
    t_1, t_2 = landmarks(log, cfg)
    return [round_iter(t_1 / 2, log), round_iter(t_1, log), round_iter(t_2, log), int(log['iter'][-1])]


def base_snapshots(log, cfg, loss_threshold=0.05):
    """Snapshot iterations of the base figure and the times they are derived from.

    Snapshots: initialization, end of alignment (t_1), end of fitting (first loss <= threshold),
    and the two midpoints, all rounded to multiples of ROUND.
    """
    t_1, t_2 = landmarks(log, cfg)
    t_end = first_iter(log, log['loss'] <= loss_threshold, int(log['iter'][-1]))
    k0, k1, k2 = 0, round_iter(t_1, log), round_iter(t_end, log)
    snapshots = [k0, round_iter((k0 + k1) / 2, log), k1, round_iter((k1 + k2) / 2, log), k2]
    return dict(t_1=t_1, t_2=t_2, t_end=t_end, snapshots=snapshots)
