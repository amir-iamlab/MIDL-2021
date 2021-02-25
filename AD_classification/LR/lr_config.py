# import the necessary packages
import os
from AD_classification.config import frequentist_config as config


# define the minimum learning rate, maximum learning rate, batch size,
# step size, CLR method, and number of epochs
MIN_LR = 1*(10**(-4))
MAX_LR = 4*(10**(-2))
STEP_SIZE = 8
CLR_METHOD = "triangular"


# define the path to the output learning rate finder plot, training
# history plot and cyclical learning rate plot
LRFIND_PLOT_PATH = os.path.sep.join([config.OUTPUT_PATH, "lrfind_plot"])
TRAINING_PLOT_PATH = os.path.sep.join([config.OUTPUT_PATH, "training_plot.png"])
CLR_PLOT_PATH = os.path.sep.join([config.OUTPUT_PATH, "clr_plot.png"])
