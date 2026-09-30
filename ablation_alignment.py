"""Ablation study, primary metric: mean alignment  cbar = (1/h) sum_j cos(w_j, sgn(v_j) u_star)
against each swept parameter, read out at the four landmark iterations of the base run."""
from common import plot_sweeps

if __name__ == '__main__':
    plot_sweeps('cbar', r'mean alignment $\bar c$', 'linear', 'ablation_alignment')
