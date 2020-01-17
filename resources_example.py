from resources import Atlas
import os
import nibabel as nib

# The atlas needs to sit in a directory, so we're using ./atlas
try:
    os.mkdir('./atlas')
except:
    pass
# Load the atlas (it will be automatically downloaded if not present)
atlas = Atlas.MNI.load('./atlas', download_if_not_found=True)

# Now we manipulate the data which are nibabel Nifti1Image types
braindata = atlas.t2.get_fdata().copy()
braindata *= atlas.mask.get_fdata()/255
brain_img = nib.Nifti1Image(braindata, atlas.t2.affine, atlas.t2.header)
nib.save(brain_img, 'brain-atlas.nii')
# You should now have a masked version of the T2 data.
