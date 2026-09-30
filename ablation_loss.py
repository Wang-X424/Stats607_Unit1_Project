"""Ablation study, secondary metric: training loss against each swept parameter,
read out at the four landmark iterations of the base run."""
from common import plot_sweeps

if __name__ == '__main__':
    plot_sweeps('loss', 'training loss', 'log', 'ablation_loss', ylim=(1e-3, 2))
