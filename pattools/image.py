from skimage.exposure import match_histograms
import numpy as np

def histogram_match(image, reference):
    '''Returns image modified to match reference's histogram'''
    return match_histograms(image, reference, multichannel=False)


def vistarsier_compare(c, p, min_val=-1., max_val=5., min_change=0.8, max_change=3.):
    """ VisTarsier's compare operation
    Parameters
    ----------
    c : ndarray
        The current volume
    p : ndarray
        The prior volume
    min_val : float
        The minimum value (measured in standard deviations) to consider
    max_val : float
        The maximum value (measured in standard deviations) to consider
    min_change : float
        The minimum change of value (measured in standard deviations) to consider
    max_change : float
        The maximum change of value (measured in standard deviations) to consider
    Returns
    -------
    change : ndarray
        The relevant change in signal.
    """
    # Get standard deviations for current and prior
    pstd = p.std()
    cstd = c.std()
    # Align prior standard deviation to current
    p = ((p - p.mean()) / pstd) * cstd + c.mean()

    #Calculate change
    change = c - p
    # Ignore change outside of minimuim and maximum values
    change[c < min_val*cstd] = 0
    change[p < min_val*cstd] = 0
    change[c > max_val*cstd] = 0
    change[p > max_val*cstd] = 0
    change[np.abs(change) < min_change*cstd] = 0
    change[np.abs(change) > max_change*cstd] = 0

    return change
