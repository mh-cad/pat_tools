import unittest
from interpolate import LinearInterpolator, _AbstractInterpolator, Renderer
import numpy as np
import nibabel as nib 
import os
import datetime
from tempfile import TemporaryDirectory
 
class TestInterpolation(unittest.TestCase):
    start_img = None
    end_img = None
    mask_img = None

    def setUp(self):
        # Create super-simple arrays to interpolate
        sdata = np.array([1, 1, 1])
        edata = np.array([28, 28, 28])
        mdata = np.array([0, 1, 0])
        self.start_img = nib.Nifti1Image(sdata, np.eye(4))
        self.end_img = nib.Nifti1Image(edata, np.eye(4))
        self.mask_img = nib.Nifti1Image(mdata, np.eye(4))

    def write_files(self, tmp_dir):
        # Save generated test files (to avoid the redundant loading time
        # for the tool pipeline).
        # We need to add the dates to the path because the interpolator 
        # will assume the date is encoded in the path. 
        # TODO: This should probably not be an assumption if the interpolator can be used
        # outside of the timelines module.
        os.mkdir(os.path.join(tmp_dir, '20200101'))
        os.mkdir(os.path.join(tmp_dir, '20200128'))
        start_path = os.path.join(tmp_dir, '20200101','start.nii')
        end_path = os.path.join(tmp_dir, '20200128', 'end.nii')
        mask_path = os.path.join(tmp_dir, 'mask.nii')
        nib.save(self.start_img, start_path)
        nib.save(self.end_img, end_path)
        nib.save(self.mask_img, mask_path)

        return start_path, end_path, mask_path

    def test_interpolated_dates(self):
        start_date = datetime.date(2020,1,1)
        end_date = datetime.date(2020,1,28)        
        interpolated_dates = _AbstractInterpolator.interpolated_dates([start_date, end_date], 1)
        # The interpolator should return 28 days
        self.assertEqual(len(interpolated_dates), 28)
        # Sanity check the data
        self.assertEqual(datetime.date(2020,1,11),interpolated_dates[10])
            
    def test_linear_interpolator(self):
        with TemporaryDirectory() as tmp_dir:
            start_path, end_path, mask_path = self.write_files(tmp_dir)
            interp = LinearInterpolator()
            # Test interpolating the data from a 2 day delta with a mask
            interpolated_data = interp.interpolated_data_from_delta((start_path, end_path), mask_path, 2)
            # Interpolated data comes as a generator, so we'll run it and get a list
            interpolated_data = list(interpolated_data)
            # Check that we have the right number of dates
            self.assertEqual(len(interpolated_data), 14)
            # Sanity check the data
            # For each entry: first value is date, second is data
            # Then we're expecting the first and last values in the data to be masked
            self.assertEqual(interpolated_data[5][0], datetime.date(2020, 1, 11))
            self.assertEqual(interpolated_data[5][1][0], 0)
            self.assertEqual(interpolated_data[5][1][1], 11)
            self.assertEqual(interpolated_data[5][1][2], 0)

            # Test for date not in current range...
            start_date = datetime.date(2020,1,1)
            end_date = datetime.date(2020,1,28)
            data = interp.data_for_date(
                datetime.date(2019,12,1), # Date is not in range
                (start_date, end_date),
                (start_path, end_path),
                1) # No mask
            
            # The output is out of range, so the expected result is zero.
            print('data:', data)
            self.assertEqual(data[1][0], data[1][1])
            self.assertEqual(data[1][1], data[1][2])
            self.assertEqual(data[1][2], 0)

            # Test for selecting closest rather than before and after (bug)
            mid_date = datetime.date(2020,1,27)
            img_mid = nib.Nifti1Image(np.array([2,2,2]), np.eye(4))
            os.mkdir(os.path.join(tmp_dir, '20200127'))
            mid_path = os.path.join(tmp_dir, '20200127','mid.nii')
            nib.save(img_mid, mid_path)

            data = interp.data_for_date(
                datetime.date(2020,1,26), # Date is not in range
                (start_date, mid_date, end_date),
                (start_path, mid_path, end_path),
                1) # No mask

            self.assertGreater(data[1][1], 1)
            self.assertLess(data[1][1], 2)
    
    def test_renderer(self):
        img = np.arange(0,64**2).reshape((64,64))
        Renderer.write_image(img, 'test.png', 0, 64**2)

if __name__ == '__main__':
    unittest.main()
    