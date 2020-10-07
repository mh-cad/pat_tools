'''A set of helper methods to access handy resources (e.g. Atlases or datasets)'''

from zipfile import ZipFile
import requests
import io
import os
import nibabel as nib
from clint.textui import progress

def download_file(url, file_path):
    r = requests.get(url, stream=True)
    with open(file_path, 'wb') as f:
        total_length = int(r.headers.get('content-length'))
        for chunk in progress.bar(r.iter_content(chunk_size=1024), expected_size=(total_length/1024) + 1):
            if chunk:
                f.write(chunk)
                f.flush()

class Atlas:
    t1 = None #T1 nifti
    t2 = None #T2 nifti
    mask = None #mask nifti
    whitematter_mask = None

    class MNI:
        mni_url = "http://www.bic.mni.mcgill.ca/~vfonov/icbm/2009/old/mni_icbm152_nlin_asym_09c_nifti.zip"

        @classmethod
        def download(self, target_dir, url = mni_url):
            print('Downloading mni_icbm152_nlin_asym_09c_nifti.zip to ', target_dir, ' from ', url)
            download_file(url, os.path.join(target_dir, 'mni_icbm152_nlin_asym_09c_nifti.zip'))
            return self.load(target_dir, download_if_not_found=False)

        @classmethod
        def load(self, target_dir, download_if_not_found=True):
            t1_path = None
            t2_path = None
            mask_path = None

            # Try to find T1 file
            if os.path.isfile(os.path.join(target_dir, 'mni_icbm152_t1_tal_nlin_asym_09c.nii')):
                t1_path = os.path.join(target_dir, 'mni_icbm152_t1_tal_nlin_asym_09c.nii')
            elif os.path.isfile(os.path.join(target_dir, 'mni_icbm152_nlin_asym_09c_nifti.zip')):
                t1_path = os.path.join(target_dir, 'mni_icbm152_t1_tal_nlin_asym_09c.nii')
                with ZipFile(os.path.join(target_dir, 'mni_icbm152_nlin_asym_09c_nifti.zip')) as z:
                    for info in z.infolist():
                        if info.filename == 'mni_icbm152_nlin_asym_09c/mni_icbm152_t1_tal_nlin_asym_09c.nii':
                            info.filename = 'mni_icbm152_t1_tal_nlin_asym_09c.nii'
                            z.extract(info, target_dir)

            # Try to find T2 file
            if os.path.isfile(os.path.join(target_dir, 'mni_icbm152_t2_tal_nlin_asym_09c.nii')):
                t2_path = os.path.join(target_dir, 'mni_icbm152_t2_tal_nlin_asym_09c.nii')
            elif os.path.isfile(os.path.join(target_dir, 'mni_icbm152_nlin_asym_09c_nifti.zip')):
                t2_path = os.path.join(target_dir, 'mni_icbm152_t2_tal_nlin_asym_09c.nii')
                with ZipFile(os.path.join(target_dir, 'mni_icbm152_nlin_asym_09c_nifti.zip')) as z:
                    for info in z.infolist():
                        if info.filename == 'mni_icbm152_nlin_asym_09c/mni_icbm152_t2_tal_nlin_asym_09c.nii':
                            info.filename = 'mni_icbm152_t2_tal_nlin_asym_09c.nii'
                            z.extract(info, target_dir)

            # Try to find Mask file
            if os.path.isfile(os.path.join(target_dir, 'mni_icbm152_t1_tal_nlin_asym_09c_mask.nii')):
                mask_path = os.path.join(target_dir, 'mni_icbm152_t1_tal_nlin_asym_09c_mask.nii')
            elif os.path.isfile(os.path.join(target_dir, 'mni_icbm152_nlin_asym_09c_nifti.zip')):
                mask_path = os.path.join(target_dir, 'mni_icbm152_t1_tal_nlin_asym_09c_mask.nii')
                with ZipFile(os.path.join(target_dir, 'mni_icbm152_nlin_asym_09c_nifti.zip')) as z:
                    for info in z.infolist():
                        if info.filename == 'mni_icbm152_nlin_asym_09c/mni_icbm152_t1_tal_nlin_asym_09c_mask.nii':
                            info.filename = 'mni_icbm152_t1_tal_nlin_asym_09c_mask.nii'
                            z.extract(info, target_dir)

            # Try to find whitematter mask file
            whitematter_path = None
            if os.path.isfile(os.path.join(target_dir, 'mni_icbm152_wm_tal_nlin_asym_09c.nii')):
                whitematter_path = os.path.join(target_dir, 'mni_icbm152_wm_tal_nlin_asym_09c.nii')
            elif os.path.isfile(os.path.join(target_dir, 'mni_icbm152_wm_tal_nlin_asym_09c.zip')):
                mask_path = os.path.join(target_dir, 'mni_icbm152_wm_tal_nlin_asym_09c.nii')
                with ZipFile(os.path.join(target_dir, 'mni_icbm152_wm_tal_nlin_asym_09c.zip')) as z:
                    for info in z.infolist():
                        if info.filename == 'mni_icbm152_wm_tal_nlin_asym_09c/mni_icbm152_wm_tal_nlin_asym_09c.nii':
                            info.filename = 'mni_icbm152_wm_tal_nlin_asym_09c.nii'
                            z.extract(info, target_dir)
            
            # Take a crack at downloading if we can't find a file
            if (t1_path == None or t2_path == None or mask_path == None or whitematter_path == None) and download_if_not_found:
                return self.download(target_dir)

            #Popualte atlas
            atlas = Atlas()
            atlas.t1 = nib.load(t1_path)
            atlas.t2 = nib.load(t2_path)
            atlas.mask = nib.load(mask_path)
            if whitematter_path != None: # Whitematter path is optional at the moment
                atlas.whitematter_mask = nib.load(whitematter_path)

            return atlas
