from utils.utils import *

TRAIN_DATA = 'data/trainData.txt'
COUNTERFEIT_COLOR = '#ff6d00cf'
GENUINE_COLOR = '#36a122cf'
COUNTERFEIT_MEAN_COLOR = '#d72b00'
GENUINE_MEAN_COLOR = '#42ff00'
ALPHA = 0.7


if __name__ == '__main__':
    logger_config()
    D, L = load(TRAIN_DATA)
    plots_dir = plot_config()
    plot_hist(D, L, plots_dir)
    plot_scatter(D, L, plots_dir)
    compute_metrics(D, L)
    