import matplotlib.pyplot as plt
import numpy as np

def plot_sos(label, prediction, path, vmin=1300, vmax=3600, exp=False):
    """
    Plot the label and prediction images side by side.

    Args:
        label (numpy.ndarray): The label image.
        prediction (numpy.ndarray): The prediction image.
        path (str): The path to save the plot.
    """
    if exp:
        label = np.log1p(label)
        prediction = np.log1p(prediction)
        vmin = np.log1p(vmin)
        vmax = np.log1p(vmax)

    _, ax = plt.subplots(1, 2, figsize=(5, 5))
    ax[0].matshow(label, cmap='jet', vmin=vmin, vmax=vmax)
    ax[0].set_title('Label')
    ax[0].axis('off')
    ax[1].matshow(prediction, cmap='jet', vmin=vmin, vmax=vmax)
    ax[1].set_title('Prediction')
    ax[1].axis('off')
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    