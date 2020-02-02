#import numpy as np
#import nibabel as nib
#import tempfile
#import os
#from dipy.align.imaffine import (transform_centers_of_mass,
#                                 AffineMap,
#                                 MutualInformationMetric,
#                                 AffineRegistration)
#from dipy.align.transforms import (TranslationTransform3D,
#                                   RigidTransform3D,
#                                   AffineTransform3D)
#
##from antspyx.util import n4_bias_field_correction
#
#def affine_registration(floating, fixed):
#    if 'ANTSPATH' in os.environ:
#        print('Using antsRegistration...')
#        with tempfile.TemporaryDirectory() as dir:
#            float_path = os.path.join(dir, 'floating.nii.gz')
#            fixed_path = os.path.join(dir, 'fixed.nii.gz')
#            nib.save(floating, float_path)
#            nib.save(fixed, fixed_path)
#            from pattools.ants import registration
#    ''' Inputs a floating and fixed image and returns the transformed floating
#     image along with the matrix '''
#
#    # Based on example @ https://dipy.org/documentation/1.1.1./examples_built/affine_registration_3d/#example-affine-registration-3d
#
#    # Load images
#    fixed_data = fixed.get_fdata()
#    floating_data = floating.get_fdata()
#
#    # Create affine map
#    identity = np.eye(4)
#    affine_map = AffineMap(
#        identity,
#        fixed_data.shape, fixed.affine,
#        floating_data.shape, floating.affine)
#
#
#    # Calculate the center of mass to give us a starting point
#    center_of_mass = transform_centers_of_mass(fixed_data, fixed.affine, floating_data, floating.affine)
#    # Create mutual information metric
#    nbins = 32 #from docs
#    sampling_prop = None # None = 100% otherwise use percentage integer. Represents number of sampled voxels
#    metric = MutualInformationMetric(nbins, sampling_prop)
#    # more settings
#    # Define the Gaussian Pyramid
#    level_iters = [10000, 1000, 100] # 3 levels of corseness with 10000x1000x100 iterations
#    sigmas = [3.0,1.0,0.0] # smoothing factors for each corseness level above
#    factors = [4,2,1] # sub-sampling at each level, in this case order of 2
#    # Now we can build an affine registration majigger
#    affreg = AffineRegistration(metric=metric,level_iters=level_iters,sigmas=sigmas,factors=factors)
#
#    # Now we're going to apply one level of transform
#    transform = TranslationTransform3D()
#    params0 = None
#    starting_affine = center_of_mass.affine
#    translation = affreg.optimize(
#        fixed_data, floating_data, transform, params0,
#        fixed.affine, floating.affine, starting_affine=starting_affine)
#    # Next, we're going to refile that shiz with a rigid transform
#    transform = RigidTransform3D()
#    params0 = None
#    starting_affine = translation.affine
#    rigid = affreg.optimize(
#        fixed_data, floating_data, transform, params0,
#        fixed.affine, floating.affine, starting_affine=starting_affine)
#    # Finally, we're going with a full affine transform.
#    transform = AffineTransform3D()
#    params0 = None
#    starting_affine = rigid.affine
#    affine = affreg.optimize(
#        fixed_data, floating_data, transform, params0,
#        fixed.affine, floating.affine, starting_affine=starting_affine)
#
#    # Build our output
#    outimg = nib.Nifti1Image(affine.transform(floating_data), floating.affine, floating.header)
#    return (outimg, affine)
#
#def n4_bias_correct(input):
#    import SimpleITK as sitk
#    with tempfile.TemporaryDirectory() as dir:
#        tmp_path = os.path.join(dir, 'n4.nii')
#        nib.save(input, tmp_path)
#        input = sitk.ReadImage(tmp_path)
#        print('        Getting Ostsu mask...')
#        mask = sitk.OtsuThreshold(input)
#        input = sitk.Cast(input, sitk.sitkFloat32)
#        print('        Running N4...')
#        n4 = sitk.N4BiasFieldCorrectionImageFilter()
#        n4.SetConvergenceThreshold=0.001
#        n4.SetBiasFieldFullWidthAtHalfMaximum=0.15
#        n4.SetMaximumNumberOfIterations=50
#        n4.SetNumberOfControlPoints=4
#        n4.SetNumberOfHistogramBins=200
#        n4.SetSplineOrder=3
#        n4.SetWienerFilterNoise=0.1
#        output = n4.Execute(input, mask)
#        sitk.WriteImage(output, tmp_path)
#        output = nib.load(tmp_path)
#        return output
#
