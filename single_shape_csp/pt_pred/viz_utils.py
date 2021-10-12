import numpy as np
import matplotlib.pyplot as plt
from matplotlib import pyplot as plt


def img_from_fig(fig, ax=None):
    """
    
    Parameters
    ----------
    fig: plt fig
        input figure
    ax: plt ax
        input axis

    Returns
    -------
    img: ndarray
        image with axis ticks removed
    """

    if ax is not None:
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        ax.set_frame_on(False)
    fig.savefig('./temp.png', pad_inches=0, bbox_inches='tight', tranparent=True)
    img = plt.imread('./temp.png')
    plt.close('all')

    return img


def viz_depths(depths):
    """
    visualize depth map along with legend

    Parameters
    ----------
    depths : ndarray [res, res]
        estimated depth map

    Returns
    -------
    depth image with the legend

    """
    depths[depths==-1] = np.max(depths) + 0.15
    fig = plt.figure(frameon=False)
    plt.imshow(depths)
    img_depth = img_from_fig(fig)
    return img_depth[:, :, :3]
