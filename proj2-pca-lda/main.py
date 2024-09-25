import numpy as np
from utils.utils import *

if __name__ == '__main__':
    logger = logger_config()
    D, L = load(config.train.path)
    plots_dir = plot_config()
    pca_problem(D, L, plots_dir)
    lda_problem(D, L, plots_dir)
    # Define a range of threshold values to test
    thresholds = np.linspace(-0.2, 0.2, 100)
    threshold, error_rate = lda_classifier(D=D, L=L)
    best_threshold, best_error_rate = find_best_threshold(D=D, L=L, thresholds=thresholds, model=lda_classifier)
    if (best_threshold, best_error_rate) != lda_classifier(D=D, L=L, threshold=best_threshold):
        logger.error("best error rate error")
        raise Exception("best error rate error")
    logger.info(f'LDA - threshold: {threshold}, error: {error_rate}')
    logger.info(f'LDA - bestThreshold: {best_threshold}, lowestError: {best_error_rate}')

    # computes error for different pca's m-values using best threshold
    for m in range(6, 0, -1):
        threshold, error_rate = pca_lda_classifier(D=D, L=L, m_pca=m)
        best_threshold, best_error_rate = find_best_threshold(D=D, L=L, thresholds=thresholds, model=pca_lda_classifier, m_pca=m)
        if (best_threshold, best_error_rate) != pca_lda_classifier(D=D, L=L, threshold=best_threshold, m_pca=m):
            logger.error("best error rate error")
            raise Exception("best error rate error")
        logger.info(f'PCA + LDA m={m} - threshold: {threshold}, error: {error_rate}')
        logger.info(f'PCA + LDA m={m} - bestThreshold: {best_threshold}, lowestError: {best_error_rate}')