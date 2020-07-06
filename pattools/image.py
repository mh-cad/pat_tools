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

def norm_z_value(img, ref_img):
    """This function will normalize the two images using z-value normalization
    Parameters
    ----------
    img : ndarray
        The image to normalize
    ref_img : ndarray
        The referene image
    Returns
    -------
    img : ndarray
        The image that's been normalized.
    """
    # Get standard deviations for current and prior
    imgstd = img.std()
    refstd = ref_img.std()
    # Align prior standard deviation to current
    img = ((img - img.mean()) / imgstd) * refstd + ref_img.mean()

    return img

def normalize_by_whitematter(img, ref_img, white_matter_mask):
    """This function will normalize two MRI brain images by histogram matching
    followed by z-value normilization on the whitematter based on a given mask.
    Parameters
    ----------
    img : ndarray
        The image to normalize
    ref_img : ndarray
        The referene image
    white_matter_mask : ndarray
        Mask where 1 in white matter and 0 is non-white matter.
    Returns
    -------
    img : ndarray
        The image that's been normalized.
    """
    # First we histogram match the whole image
    img = histogram_match(img, ref_img)
    # Then we're going to perform z-score normalisation using the whitematter
    # masked means and std deviation. This should get the whitematter values
    # as close as possible.
    print('type(ref_img)', type(ref_img))
    print('type(img)', type(img))
    print('type(white_matter_mask)', type(white_matter_mask))
    masked_ref = ref_img * white_matter_mask
    masked_img = img * white_matter_mask
    mrstd = masked_ref.std()
    mistd = masked_img.std()
    normed_img = ((img - masked_img.mean()) / mistd) * mrstd + masked_ref.mean()

    return normed_img