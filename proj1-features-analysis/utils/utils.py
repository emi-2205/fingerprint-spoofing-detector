import numpy as np
import logging
import matplotlib.pyplot as plt # type: ignore
from matplotlib.backends.backend_pdf import PdfPages # type: ignore
from omegaconf import OmegaConf # type: ignore
import os

# Load the YAML configuration file
config = OmegaConf.load("config/default.yaml")

def logger_config():
    # Set up basic configuration
    logging.basicConfig(
        level=logging.INFO,  # Set the minimum logging level for your application
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('proj1-features-analysis/app.log', mode='w'),  # Log to a file
            logging.StreamHandler()  # Also log to the console
        ]
    )
    # Set the logging level for Matplotlib to WARNING to suppress DEBUG and INFO messages
    logging.getLogger('matplotlib').setLevel(logging.WARNING)

def vcol(v):
    return v.reshape((1, v.size))

def vrow(v): # or mcol
    return v.reshape((v.size, 1))

def load(file):
    # Create a logger
    logger = logging.getLogger(__name__)
    D = []  #data/features
    L = []  #labels
    with open(file) as f:
        for line in f:
            try:
                features = line.split(',')[0:-1]
                sample = vrow(np.array([float(i) for i in features]))
                label = line.split(",")[-1].strip()
                D.append(sample)
                L.append(label)
            except:
                logger.error('error loading data')
    return np.hstack(D), np.array(L, dtype=np.int32)

def plot_config():
    plt.rc('font', size=16)
    plt.rc('xtick', labelsize=16)
    plt.rc('ytick', labelsize=16)
    # Create the plots directory relative to the current file's location
    plots_dir = os.path.join(os.path.dirname(__file__), '../plots')
    os.makedirs(plots_dir, exist_ok=True)
    return plots_dir

def plot_hist(D, L, plots_dir):
    D0 = D[:, L==0]
    D1 = D[:, L==1]
    # Use the absolute path for saving the PDF
    pdf_path = os.path.join(plots_dir, 'hist.pdf')
    # Open PdfPages with the correct file path
    with PdfPages(pdf_path) as pdf:
        for dIdx in range(6):
            plt.figure()
            plt.xlabel(f"feature_{dIdx + 1}")
            plt.hist(D0[dIdx, :], bins=10, color=config.counterfeit.color, density=True, alpha=config.counterfeit.alpha, label='Counterfeit')
            plt.hist(D1[dIdx, :], bins=10, color=config.genuine.color, density=True, alpha=config.genuine.alpha, label='Genuine')
            plt.legend()
            plt.tight_layout()
            pdf.savefig()  # Save the current figure to the pdf
            plt.close()    # Close the figure

def plot_scatter(D, L, plots_dir):
    D0 = D[:, L==0]
    D1 = D[:, L==1]
    # Use the absolute path for saving the PDF
    pdf_path = os.path.join(plots_dir, 'scatter.pdf')
    with PdfPages(pdf_path) as pdf:
        for dIdx1 in range(1, 6, 2): 
            for dIdx2 in range(0, 5, 2):
                if not(dIdx1 > dIdx2 and dIdx1-dIdx2==1):
                    continue
                plt.figure()
                plt.xlabel(f"feature_{dIdx1+1}")
                plt.ylabel(f"feature_{dIdx2+1}")
                plt.scatter(D0[dIdx1, :], D0[dIdx2, :], color= config.counterfeit.color, alpha = 0.4, label = 'Counterfeit')
                plt.scatter(D1[dIdx1, :], D1[dIdx2, :], color= config.genuine.color, alpha = 0.4, label = 'Genuine')
            
                mean_D0 = np.mean(D0[[dIdx1, dIdx2], :], axis=1)
                mean_D1 = np.mean(D1[[dIdx1, dIdx2], :], axis=1)
                plt.scatter(mean_D0[0], mean_D0[1], color= config.counterfeit.mean_color, marker="x", zorder=10)
                plt.scatter(mean_D1[0], mean_D1[1], color= config.genuine.mean_color, marker="x", zorder=10)

                plt.legend()
                plt.tight_layout() # Use with non-default font size to keep axis label inside the figure
                pdf.savefig()
                plt.close()

def compute_metrics(D, L):
    logger = logging.getLogger(__name__)
    D0 = D[:, L==0]
    D1 = D[:, L==1]
    for dIdx in range(6):
            mu = D0[dIdx:dIdx+1, :].mean(1).reshape((D0[dIdx:dIdx+1, :].shape[0], 1))
            logger.info(f'Mean D0, F{dIdx+1}: {mu}')
            mu = D1[dIdx:dIdx+1, :].mean(1).reshape((D1[dIdx:dIdx+1, :].shape[0], 1))
            logger.info(f'Mean D1, F{dIdx+1}: {mu}')

    var = D0.var(1)
    std = D0.std(1)
    logger.info(f'Variance D0: {var}')
    logger.info(f'Std. dev. D0: {std}')
    
    var = D1.var(1)
    std = D1.std(1)
    logger.info(f'Variance D1: {var}')
    logger.info(f'Std. dev. D1: {std}')