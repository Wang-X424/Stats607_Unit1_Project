"""Base experiment (Fig. 2): five snapshots of the filters projected onto the signal plane
(initialization, end of alignment, after fitting, and the two midpoints) and, separately, the
loss curve with the measured t_1 and t_2. The snapshots come in two versions: filters
normalized to unit norm before projection (directions only) and raw projections (norms visible)."""
import os
import numpy as np
import matplotlib.pyplot as plt
from common import base_run, landmarks, first_iter, round_iter, class_angle, BASE, FIGS

COLORS = {'pos_data': 'tab:blue', 'neg_data': 'tab:orange',
          'pos_filter': 'tab:green', 'neg_filter': 'tab:red', 'dead': 'lightgrey'}
# filled markers with a thin black edge so that filters stay visible on top of the data points
FILTER_MARKER = {True: dict(marker='o', ms=5, mec='k', mew=0.4, ls=''),     # v_j > 0
                 False: dict(marker='X', ms=6, mec='k', mew=0.4, ls='')}    # v_j < 0


def draw_plane(ax, log, i, normalized, lim):
    """Signal patches on the unit circle, the class axes u_+ and u_- (normalized sums of the
    positive / negative signal patches, drawn with unit length), and one marker per filter at
    P_sig(w_j / ||w_j||) if normalized else P_sig(w_j), at log index i. Dots for v_j > 0,
    crosses for v_j < 0; filters that activate no signal patch (frozen for good) in grey."""
    s, y = log['signals'], log['y']
    ax.plot(*s[y > 0].T, 'o', ms=5, mfc='none', color=COLORS['pos_data'], label='positive signal patch')
    ax.plot(*s[y < 0].T, 'x', ms=5, color=COLORS['neg_data'], label='negative signal patch')
    for cls, key, label in [(y > 0, 'pos_filter', '$u_+$'), (y < 0, 'neg_filter', '$u_-$')]:
        u = s[cls].sum(0) / np.linalg.norm(s[cls].sum(0))
        ax.plot([0, u[0]], [0, u[1]], '-', lw=1.5, color=COLORS[key], label=label)
    proj, sgn, alive = log['proj'][i], log['sgn'][i], log['alive'][-1]
    if normalized:
        proj = proj / log['norm'][i][:, None]
    for p, s_, a in zip(proj, sgn, alive):
        color = COLORS['dead'] if not a else COLORS['pos_filter'] if s_ > 0 else COLORS['neg_filter']
        ax.plot(p[0], p[1], **FILTER_MARKER[s_ > 0], color=color)
    ax.set(xlim=(-lim, lim), ylim=(-lim, lim), aspect='equal')


def main():
    log = base_run()
    t_align, t2 = landmarks(log)
    t_end = first_iter(log, log['loss'] <= 0.05, int(log['iter'][-1]))
    k0, k1, k2 = 0, round_iter(t_align, log), round_iter(t_end, log)
    # k0 = 0
    # k1 = 
    # k2 =
    snapshots = [k0, round_iter((k0 + k1) / 2, log), k1, round_iter((k1 + k2) / 2, log), k2]
    idx = lambda k: int(np.searchsorted(log['iter'], k))

    # Size the figure close to its final ICLR single-column width.
    # This prevents LaTeX from shrinking a very wide canvas and making text tiny.
    paper_width = 6.9

    for normalized, tag in [(True, 'normalized'), (False, 'raw')]:
        lim = 1.3 if normalized else max(
            1.3, 1.1 * np.abs(log['proj'][idx(snapshots[-1])]).max()
        )

        fig, axes = plt.subplots(
            1, len(snapshots),
            figsize=(paper_width, 2.15),
            sharex=True, sharey=True,
            gridspec_kw={'wspace': 0.08},
        )

        for col, (ax, k, letter) in enumerate(zip(axes, snapshots, 'abcde')):
            draw_plane(ax, log, idx(k), normalized, lim)

            # Keep titles compact; the caption already explains that these are iterations.
            ax.set_title(rf'({letter}) $iter={k}$', fontsize=8, pad=2)

            # Sparse ticks remain readable at final paper size.
            if normalized:
                ax.set_xticks([-1.0, 0.0, 1.0])
                ax.set_yticks([-1.0, 0.0, 1.0])

            ax.tick_params(axis='both', which='major', labelsize=7, pad=1.5)

            # All panels use the same y-scale, so only the first needs y tick labels.
            if col > 0:
                ax.tick_params(labelleft=False)

        # Add the filter legend entries once, then use one shared legend.
        for key, positive, label in [
            ('pos_filter', True, r'filter, $v_j>0$'),
            ('neg_filter', False, r'filter, $v_j<0$'),
            ('dead', False, 'filter, no signal patch active'),
        ]:
            axes[0].plot(
                [], [], **FILTER_MARKER[positive],
                color=COLORS[key], label=label
            )

        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(
            handles, labels,
            loc='lower center',
            bbox_to_anchor=(0.5, 0.01),
            ncol=4,
            fontsize=7.2,
            frameon=False,
            handlelength=1.4,
            handletextpad=0.4,
            columnspacing=1.0,
        )

        # Leave room for the shared legend while keeping the five panels large.
        fig.subplots_adjust(
            left=0.055, right=0.995, top=0.89, bottom=0.28
        )

        fig.savefig(
            os.path.join(FIGS, f'base_experiment_{tag}.pdf'),
            bbox_inches='tight',
            pad_inches=0.02,
        )
        fig.savefig(
            os.path.join(FIGS, f'base_experiment_{tag}.png'),
            dpi=300,
            bbox_inches='tight',
            pad_inches=0.02,
        )

    fig, ax = plt.subplots(figsize=(4.8, 2.8))
    ax.plot(log['iter'], log['loss'], color='tab:blue')
    for k, name in [(t_align, r'$\hat t_{\rm align}$'), (t2, r'$\hat t_2$')]:
        ax.axvline(k, color='k', ls='--', lw=0.8)
        ax.annotate(f'{name} = {k}', (k, 1.02), rotation=90, va='bottom', ha='right', fontsize=9)
    ax.set(xlabel='iteration', ylabel='training loss', ylim=(0, 1.35))
    ax.tick_params(axis='both', labelsize=8)
    ax.xaxis.label.set_size(9)
    ax.yaxis.label.set_size(9)
    fig.tight_layout()
    fig.savefig(
        os.path.join(FIGS, 'base_experiment_loss.pdf'),
        bbox_inches='tight',
        pad_inches=0.02,
    )
    fig.savefig(
        os.path.join(FIGS, 'base_experiment_loss.png'),
        dpi=300,
        bbox_inches='tight',
        pad_inches=0.02,
    )

    alive = log['alive'][-1]
    t_S = first_iter(log, log['fracS'] >= alive.mean(), None)     # every live filter in S_+/S_-
    print(f't_align = {t_align} (all live filters inside their class cone); every live filter in S+/S- '
          f'from iteration {t_S}; t_2 = {t2}; loss <= 0.05 from {t_end}; snapshots at {snapshots}; '
          f'final loss = {log["loss"][-1]:.3g}; '
          f'{(~alive).sum()} of {len(alive)} filters activate no signal patch; '
          f'final angle to class axis, live filters: max {class_angle(log, BASE)[-1, alive].max():.1f} deg')


if __name__ == '__main__':
    main()
