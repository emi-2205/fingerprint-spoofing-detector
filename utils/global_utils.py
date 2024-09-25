import numpy as np
import logging

def logger_config():
    # Set up basic configuration
    logging.basicConfig(
        level=logging.DEBUG,  # Set the minimum logging level
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',  # Log format
        handlers=[
            logging.FileHandler('app.log'),  # Log to a file
            logging.StreamHandler()  # Also log to the console
        ]
    )

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