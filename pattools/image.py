from skimage.exposure import match_histograms

def histogram_match(image, reference):
    '''Returns image modified to match reference's histogram'''
    return match_histograms(image, reference, multichannel=False)
