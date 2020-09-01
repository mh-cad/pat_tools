from skimage.exposure import match_histograms
from sklearn.mixture import GaussianMixture
import numpy as np

def histogram_match(image, reference):
    '''Returns image modified to match reference's histogram'''
    return match_histograms(image, reference, multichannel=False)

def _build_gaussian_mixture(data, n_components, random_sample=100000):
    '''Builds and fits a GaussianMixture from a random sample of the provided data'''
    gm = GaussianMixture(n_components=n_components, random_state=42069)
    rng = np.random.default_rng(42069) # We're seeding our RNG to make it deterministic
    gm.fit(rng.choice(data.reshape(-1), random_sample).reshape(-1,1))
    return gm

def _find_key_intesities(data, mixture_model):
    ''' Finds the key intensity values for scaling based on the mixture model.'''
    max_val = np.max(data)
    if max_val == 0:
        return [0, 0]
    dist = mixture_model.predict_proba(np.arange(max_val).reshape(-1,1))
    # Find the ordered distribution indexes
    ordered_idx = []
    idx = -1
    for d in dist:
        imax = np.argmax(d)
        if idx != imax:
            idx = imax
            ordered_idx.append(idx)
    # Find the key intensity values
    keys = []
    # Minimum value is the first key (should usually be zero).
    keys.append(np.min(data))
    # For each distribution within the model where oi is the index into 
    # ordered idx and di is the index into the dist entries.
    # We're ignoring the last distribution because the peak is poorly defined
    # and there's no next crossover point.
    for oi, di in enumerate(ordered_idx[:-1]):
        # Add the peak value
        peak = np.argmax(dist[:,di])
        keys.append(peak)
        # Find the cross over point
        # Check for the first value from the peak where the next distribution
        # becomes more probable than the current one. Add this as a key point.
        di_next = ordered_idx[oi+1]
        i = peak
        while (i < max_val and di_next < len(dist[i])):
            if dist[i][di] <= dist[i][di_next]:
                keys.append(i)
                break
            i+=1
    # Finally add the max value and return.
    keys.append(max_val)
    return np.array(keys)

def mixed_model_match(image, reference, n_components=5):
    # Build gaussian mixture models
    ref_gm = _build_gaussian_mixture(reference, n_components)
    img_gm = _build_gaussian_mixture(image, n_components)
    # Find the key intensity values for scaling
    ref_keys = _find_key_intesities(reference, ref_gm)
    img_keys = _find_key_intesities(image, img_gm)

    # We're going to use bias as the minimum value and scale as max
    # This will mean that we should end up with the same min and max vals
    #ref_bias = ref_keys[0]
    #ref_scale = ref_keys[-1] - ref_bias
    #img_bias = img_keys[0]
    #img_scale = img_keys[-1] - img_bias
    ## Apply the operation to both the image and the keys
    #image = (image - img_bias) / img_scale * ref_scale + ref_bias
    #img_keys = (img_keys - img_bias) / img_scale * ref_scale + ref_bias
    # Get a set of intervals
    img_intervals = list(zip(img_keys, img_keys[1:]))
    ref_intervals = list(zip(ref_keys, ref_keys[1:]))
    # We want to calculate the masks before we change thing.
    # Although it will be a monotocally increasing set there may be overlap in the process.
    interval_masks = []
    for istart, iend in img_intervals:    
        interval_masks.append(np.logical_and(image > istart, image <= iend))
    interval_masks.reverse() #this allows us to pop()
    # Now we scale each interval individually
    for (istart, iend), (rstart, rend) in zip(img_intervals, ref_intervals):
        interval_mask = interval_masks.pop()
        rbias = rstart
        rscale = rend - rstart
        ibias = istart
        iscale = iend - istart
        image[interval_mask] -= ibias
        print('ibias, rbias',ibias, rbias)
        if iscale != 0:
            image[interval_mask] /= iscale
        image[interval_mask] = image[interval_mask] * rscale + rbias
    # ...and done!
    return image

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
    masked_ref = ref_img * white_matter_mask
    masked_img = img * white_matter_mask
    mrstd = masked_ref.std()
    mistd = masked_img.std()
    normed_img = ((img - masked_img.mean()) / mistd) * mrstd + masked_ref.mean()

    return normed_img

def estimate_window(data: np.ndarray):
    '''Estimates a viewing window based on the given numpy array. Just using mean and std but it's a start'''
    wl = np.mean(data[data > 20])
    ww = 3 * np.std(data[data > 20])
    
    if wl == np.nan or wl == np.Infinity: wl = 0
    if ww == np.nan or ww == np.Infinity: ww = 0
    return (wl, ww)

# This function was stolen from a numpy cookbook and provides smoothing as is 
# extrememly well commented (and then expressed through some very dense code with single letter vars). 
def smooth(x, window_len=11, window='hanning'):
    """smooth the data using a window with requested size.
    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal 
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.
    input:
        x: the input signal 
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.
    output:
        the smoothed signal
    example:
    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)
    see also: 
    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter
    TODO: the window parameter could be the window itself if an array instead of a string
    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
    """
    if x.ndim != 1:
        raise ValueError("smooth only accepts 1 dimension arrays.")
    if x.size < window_len:
        raise ValueError("Input vector needs to be bigger than window size.")

    if window_len<3:
        return x

    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError("Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")

    s=np.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]]
    #print(len(s))
    if window == 'flat': #moving average
        w=np.ones(window_len,'d')
    else:
        w=eval('np.'+window+'(window_len)')
    y=np.convolve(w/w.sum(),s,mode='valid')
    return y