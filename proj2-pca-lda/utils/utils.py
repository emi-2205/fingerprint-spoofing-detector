import numpy as np
import logging
import matplotlib.pyplot as plt # type: ignore
from matplotlib.backends.backend_pdf import PdfPages # type: ignore
from omegaconf import OmegaConf # type: ignore
import os
import scipy.linalg # type: ignore

# Load the YAML configuration file
config = OmegaConf.load("config/default.yaml")

def logger_config():
    # Set up basic configuration
    logging.basicConfig(
        level=logging.INFO,  # Set the minimum logging level for your application
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('proj2-pca-lda/app.log', mode='w'),  # Log to a file
            logging.StreamHandler()  # Also log to the console
        ]
    )
    # Set the logging level for Matplotlib to WARNING to suppress DEBUG and INFO messages
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    return  logging.getLogger(__name__)

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
                raise Exception("Error loading data")
    return np.hstack(D), np.array(L, dtype=np.int32)

def plot_config():
    plt.rc('font', size=16)
    plt.rc('xtick', labelsize=16)
    plt.rc('ytick', labelsize=16)
    # Create the plots directory relative to the current file's location
    plots_dir = os.path.join(os.path.dirname(__file__), '../plots')
    os.makedirs(plots_dir, exist_ok=True)
    return plots_dir

def plot_hist(D, L, fname, plots_dir, m):
    D0 = D[:, L==0]
    D1 = D[:, L==1]
    # Use the absolute path for saving the PDF
    pdf_path = os.path.join(plots_dir, f'hist_{fname}_m={m}.pdf')
    # Open PdfPages with the correct file path
    with PdfPages(pdf_path) as pdf:
        for dIdx in range(m):
            plt.figure()
            plt.xlabel(f"feature_{dIdx + 1}")
            plt.hist(D0[dIdx, :], bins=10, color=config.counterfeit.color, density=True, alpha=config.counterfeit.alpha, label='Counterfeit')
            plt.hist(D1[dIdx, :], bins=10, color=config.genuine.color, density=True, alpha=config.genuine.alpha, label='Genuine')
            plt.legend()
            plt.tight_layout()
            pdf.savefig()  # Save the current figure to the pdf
            plt.close()    # Close the figure

def plot_scatter(D, L, fname, plots_dir, m):
    logger = logging.getLogger(__name__)
    if(m==1):
        logger.error("m value must be >1")
        raise Exception("Parameter Value Error")
    
    D0 = D[:, L==0]
    D1 = D[:, L==1]
    # Use the absolute path for saving the PDF
    pdf_path = os.path.join(plots_dir, f'scatter_{fname}_m={m}.pdf')
    with PdfPages(pdf_path) as pdf:
        for dIdx1 in range(1, m, 2): 
            for dIdx2 in range(0, m-1, 2):
                if not(dIdx1 > dIdx2 and dIdx1-dIdx2==1):
                    continue
                plt.figure()
                plt.xlabel(f"feature_{dIdx1+1}")
                plt.ylabel(f"feature_{dIdx2+1}")
                plt.scatter(D0[dIdx1, :], D0[dIdx2, :], color=config.counterfeit.color, alpha = 0.4, label = 'Counterfeit')
                plt.scatter(D1[dIdx1, :], D1[dIdx2, :], color=config.genuine.color, alpha = 0.4, label = 'Genuine')
            
                mean_D0 = np.mean(D0[[dIdx1, dIdx2], :], axis=1)
                mean_D1 = np.mean(D1[[dIdx1, dIdx2], :], axis=1)
                plt.scatter(mean_D0[0], mean_D0[1], color=config.counterfeit.mean_color, marker="x", zorder=10)
                plt.scatter(mean_D1[0], mean_D1[1], color=config.genuine.mean_color, marker="x", zorder=10)

                plt.legend()
                plt.tight_layout() # Use with non-default font size to keep axis label inside the figure
                pdf.savefig()
                plt.close()

def pca(D, m):
    mu = vrow(D.mean(1)) # asse 1 colonne
    DC = D - mu # np broadcasting
    N = DC.shape[1] # D.size restituisce il prodotto delle dimensioni, occhio!!
    C = (DC@DC.T)/N
    s, U = np.linalg.eigh(C) # return lambdas in decreasing order
    P = U[:, ::-1][:, 0:m] 
    DP = P.T@D
    return DP, P

def pca_problem(D, L, plots_dir, m=6):
    DP, P = pca(D, m)
    plot_hist(DP, L, "PCA", plots_dir, m)
    plot_scatter(DP, L, "PCA", plots_dir, m)

def compute_Sb_Sw(D, L):
    Sb = 0
    Sw = 0
    muGlobal = vrow(D.mean(1))  ##? dataset mean
    for i in np.unique(L):  # iterate over each class (0,1,2)
        DCls = D[:, L == i]  # select sample of class i
        mu = vrow(DCls.mean(1))  ##? class mean
        Sb += (mu - muGlobal) @ (mu - muGlobal).T * DCls.shape[1]  # between
        Sw += (DCls - mu) @ (DCls - mu).T  # within
    return Sb / D.shape[1], Sw / D.shape[1]

def generalized(SB, SW, m):
    s, U = scipy.linalg.eigh(SB, SW)
    W = U[:, ::-1][:, 0:m]
    return W
    
def lda(D, L, m, left_label, right_label):
    SB, SW = compute_Sb_Sw(D, L)
    W = generalized(SB, SW, m)
    DP = W.T@D
    # Check if the mean of the right_label is greater than the left_label
    if DP[0, L == right_label].mean() < DP[0, L == left_label].mean():
        # Flip the sign of the projection if it's the wrong way
        DP = -DP
        W = -W
    return DP, W

def lda_problem(D, L, plots_dir, m=6):
    DP, W = lda(D, L, m, 0, 1)
    plot_scatter(DP, L, "LDA", plots_dir, m)
    plot_hist(DP, L, "LDA", plots_dir, m)

def split_db_2to1(D, L, seed=0):
    nTrain = int(D.shape[1]*2.0/3.0) 
    np.random.seed(seed)
    idx = np.random.permutation(D.shape[1]) 
    idxTrain = idx[0:nTrain]
    idxTest = idx[nTrain:]
    DTR = D[:, idxTrain]
    DVAL = D[:, idxTest]
    LTR = L[idxTrain]
    LVAL = L[idxTest]
    return (DTR, LTR), (DVAL, LVAL)

def find_best_threshold(D, L, thresholds, model, m_pca= None):
    best_threshold = None
    best_error_rate = float('inf')

    for threshold in thresholds:
        if m_pca == None:
            _, error_rate = model(D, L, threshold)
        else:
            _, error_rate = model(D, L, threshold, m_pca)
    
        if error_rate < best_error_rate:
            best_error_rate = error_rate
            best_threshold = threshold

    return best_threshold, best_error_rate

#* LDA
def lda_classifier(D, L, threshold=None, m_lda=1):
    (DTR, LTR), (DVAL, LVAL) = split_db_2to1(D, L)
    DTR_lda, W = lda(DTR, LTR, m_lda, 0, 1)
    DVAL_lda = W.T@DVAL
    if threshold==None:
        threshold = (DTR_lda[0, LTR == 0].mean()+DTR_lda[0, LTR == 1].mean())/2.0
    PVAL = np.zeros(shape=LVAL.shape, dtype=np.int32) 
    PVAL[DVAL_lda[0] >= threshold] = 1
    PVAL[DVAL_lda[0] < threshold] = 0

    error_rate = (np.count_nonzero(PVAL-LVAL)/DVAL_lda.shape[1])*100
    return threshold, error_rate

#* PCA + LDA
def pca_lda_classifier(D, L, threshold=None, m_pca=6, m_lda=1):
    (DTR, LTR), (DVAL, LVAL) = split_db_2to1(D, L)
    DTR_pca, P = pca(DTR, m_pca)
    DVAL_pca = P.T@DVAL # PCA estimated on the model training data only -> we use P
    DTR_lda, W = lda(DTR_pca, LTR, m_lda, 0, 1)
    DVAL_lda = W.T@DVAL_pca
    if threshold==None:
        threshold = (DTR_lda[0, LTR == 0].mean()+DTR_lda[0, LTR == 1].mean())/2.0
    PVAL = np.zeros(shape=LVAL.shape, dtype=np.int32) 
    PVAL[DVAL_lda[0] >= threshold] = 1
    PVAL[DVAL_lda[0] < threshold] = 0

    error_rate = (np.count_nonzero(PVAL-LVAL)/DVAL_lda.shape[1])*100
    return threshold, error_rate