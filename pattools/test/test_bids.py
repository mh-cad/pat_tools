import unittest
from tempfile import TemporaryDirectory
from pattools.bids import *
import os

class TestDataset(unittest.TestCase):
    
    def setUp(self):
        pass

    def create_folders(self, tmpdir):
        os.makedirs(os.path.join(tmpdir, 'sub-01', 'ses-01', 'anat'))
        os.makedirs(os.path.join(tmpdir, 'sub-02', 'anat'))
        

    def test_read(self):
        with TemporaryDirectory() as tmpdir:
            self.create_folders(tmpdir)

            ds = Dataset(tmpdir)
            subs = ds.subjects()
        
            assert(subs[0].label == '01')
            assert(subs[1].label == '02')
            assert(subs[0].sessions()[0].label == '01')
            assert(subs[1].sessions()[0].label == None)
