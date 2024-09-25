from utils.utils import *


if __name__ == '__main__':
    config = OmegaConf.load("config/default.yaml")
    logger_config()
    D, L = load(config.train.path)
    plots_dir = plot_config()
    plot_hist(D, L, plots_dir)
    plot_scatter(D, L, plots_dir)
    compute_metrics(D, L)
    