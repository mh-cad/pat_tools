import unittest
from tempfile import TemporaryDirectory
from pattools.bids import *
import os
import json

class TestDataset(unittest.TestCase):
    
    test_data = '''
    {
        "Name": "FMRIPREP Outputs",
        "BIDSVersion": "1.4.0",
        "DatasetType": "derivative",
        "GeneratedBy": [
            {
            "Name": "fmriprep",
            "Version": "1.4.1",
            "Container": {
                "Type": "docker",
                "Tag": "poldracklab/fmriprep:1.4.1"
                }
            },
            {
            "Name": "Manual",
            "Description": "Re-added RepetitionTime metadata to bold.json files"
            }
        ],
        "SourceDatasets": [
            {
            "DOI": "doi:10.18112/openneuro.ds000114.v1.0.1",
            "URL": "https://openneuro.org/datasets/ds000114/versions/1.0.1",
            "Version": "1.0.1"
            }
        ]
        }
    '''

    def setUp(self):
        pass

    def create_folders(self, tmpdir):
        os.makedirs(os.path.join(tmpdir, 'sub-01', 'ses-01', 'anat'))
        os.makedirs(os.path.join(tmpdir, 'sub-02', 'anat'))
        with open(os.path.join(tmpdir, 'sub-01', 'ses-01', 'anat', 'sub-01_ses-01_test2.nii'), 'w') as f:
            f.write('totally real image')
        with open(os.path.join(tmpdir, 'sub-02', 'anat','sub-02_test1.nii.gz'), 'w') as f:
            f.write('totally real image')

    def test_read(self):
        with TemporaryDirectory() as tmpdir:
            self.create_folders(tmpdir)

            ds = Dataset(tmpdir)
            subs = ds.subjects()
        
            assert(subs[0].label == '01')
            assert(subs[1].label == '02')
            assert(subs[0].sessions()[0].label == '01')
            assert(subs[1].sessions()[0].label == None)

            scan1 = subs[0].sessions()[0].modalities()[0].scans()[0]
            scan2 = subs[1].sessions()[0].modalities()[0].scans()[0]
            scan1.write_metadata({'foo':'bar'})
            assert(scan1.read_metadata()['foo'] == 'bar')

            assert(scan1.path == os.path.join(tmpdir, 'sub-01', 'ses-01', 'anat', 'sub-01_ses-01_test2.nii'))

    def test_description(self):
        ddct = json.loads(self.test_data)
        assert(ddct['Name'] == 'FMRIPREP Outputs')
        dd = DatasetDescription.from_dict(ddct)
        dct = dd.to_dict()
        assert(dct['Name'] == 'FMRIPREP Outputs')

        assert(dd.SourceDatasets[0].DOI == "doi:10.18112/openneuro.ds000114.v1.0.1")

        with TemporaryDirectory() as tmpdir:
            dd.path = os.path.join(tmpdir, 'dataset_description.json')
            dd.save()
            loaded = DatasetDescription.load(dd.path)
            assert(loaded.GeneratedBy[1].Name == "Manual")
