from tempfile import TemporaryDirectory
from pattools.pacs import Patient
from pattools._timelinefilter import flair_filter
from pattools.ants import n4_bias_correct, affine_registration
from pattools.fsl import bet
import os
import subprocess
import numpy as np

def latest_flair_pair(patient_id, scp_settings, tmp_dir):
    ''' Get the 2 most recent FLAIR studies for a patient. Returns None if we can't find 2 studies'''
    p = Patient(os.path.join(tmp_dir, patient_id), scp_settings=scp_settings)
    studies = p.find_studies()
    studies.sort(key=lambda s: s.study_date, reverse=True)
    fil = flair_filter()
    latest_flairs = []
    for s in studies:
        flair = fil.filter(s)
        if flair != None:
            latest_flairs.append(flair)
        if len(latest_flairs) >= 2:
            break
    if len(latest_flairs) == 2:
        return latest_flairs
    else:
        return None

def preprocess_pair(floating, fixed, tmp_dir):
    """Pre processes the nifti images using:
    ANTs N4BiasFieldCorrection -> FSL bet -> ANTs antsRegistration
    Parameters
    ----------
    floating : string
        Path to the floating nifti image
    fixed : string
        Path to the fixed nifti image
    Returns
    -------
    (floating, fixed) : tuple
        floating : string
            Path to the pre-processed floating image
        fixed : string
            Path to the pre-processed floating image
    """
    with TemporaryDirectory() as local_temp:
        # ANTS N4 Bias correction
        outfloat = os.path.join(local_temp, 'n4float.nii.gz')
        outfixed = os.path.join(local_temp, 'n4fixed.nii.gz')
        p1 = n4_bias_correct(floating, outfloat)
        p2 = n4_bias_correct(fixed, outfixed)
        p1.wait()
        p2.wait()
        # FSL BET
        floating = outfloat
        fixed = outfixed
        outfloat = os.path.join(local_temp, 'betfloat.nii.gz')
        outfixed = os.path.join(tmp_dir, 'fixed-processed.nii.gz') # Registration won't change this
        p1 = bet(floating, outfloat)
        p2 = bet(fixed, outfixed)
        p1.wait()
        p2.wait()
        # Coregistration
        floating = outfloat
        fixed = outfixed
        outfloat = os.path.join(tmp_dir, 'floating-processed.nii.gz')
        affine_registration(floating, fixed, outfloat).wait()
        
        return outfloat, outfixed

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