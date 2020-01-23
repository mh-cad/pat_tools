from pattools.pacs import Series, Patient
from pattools.resources import Atlas
import nibabel as nib
import multiprocessing
import json
import os
import shutil
from clint.textui import progress
from tempfile import TemporaryDirectory


def affine_registration(input):
    pass
class ScorecardElement:
    description = None
    points = 0
    def __init__(self, description, points):
        self.description = description
        self.points = points

class Filter:
    name = None
    scorecard = None # For matching description
    min_rows = 100
    min_cols = 100
    threshold = 0

    def __init__(self, name):
        self.name = name
        self.scorecard = []

    def filter(self, study):
        series = study.find_series()
        candidates = []

        for seri in series:
            score = 0
            # If the description contains a string on the scorecard..
            for se in self.scorecard:
                if se.description.lower() in seri.description.lower():
                    score += se.points #... add to the score
            if score >= self.threshold:
                candidates.append((score, seri))
        # Sort the candidates by score
        candidates.sort(key=lambda x: x[0], reverse=True)
        # If theres a candidate, append the best
        if len(candidates) > 0:
            return candidates[0][1]
        # No valid series found
        return None


def flair_filter():
    filter = Filter('FLAIR')
    filter.scorecard.append(ScorecardElement('flair', 100))
    filter.scorecard.append(ScorecardElement('mprage', -100))
    filter.scorecard.append(ScorecardElement('localiser', -50))
    filter.scorecard.append(ScorecardElement('sag', 20))
    filter.scorecard.append(ScorecardElement('_cor', -5))
    filter.scorecard.append(ScorecardElement('_tra', -5))
    filter.scorecard.append(ScorecardElement('_cor', -5))
    filter.scorecard.append(ScorecardElement('t2', 45))
    filter.scorecard.append(ScorecardElement('t1', -45))
    filter.scorecard.append(ScorecardElement('3d', 25))
    filter.scorecard.append(ScorecardElement('dis3d', -25))
    filter.scorecard.append(ScorecardElement('report', -50))
    filter.scorecard.append(ScorecardElement('fused', -35))
    filter.threshold = 100
    return filter

def mprage_filter():
    filter = Filter('MPRAGE')
    filter.scorecard.append(ScorecardElement('mprage', 100))
    filter.scorecard.append(ScorecardElement('flair', -100))
    filter.scorecard.append(ScorecardElement('localiser', -50))
    filter.scorecard.append(ScorecardElement('sag', 20))
    filter.scorecard.append(ScorecardElement('_cor', -5))
    filter.scorecard.append(ScorecardElement('_tra', -5))
    filter.scorecard.append(ScorecardElement('_cor', -5))
    filter.scorecard.append(ScorecardElement('t1', 30))
    filter.scorecard.append(ScorecardElement('t2', -30))
    filter.scorecard.append(ScorecardElement('3d', 25))
    filter.scorecard.append(ScorecardElement('dis3d', -25))
    filter.scorecard.append(ScorecardElement('c+', -5))
    filter.scorecard.append(ScorecardElement('report', -50))
    filter.scorecard.append(ScorecardElement('fused', -35))
    filter.threshold = 100
    return filter

def default_filters():
    return [flair_filter(), mprage_filter()]

class FileMetadata:
    file = None
    processed_file = None
    filter_name = None
    study_uid = None
    series_uid = None
    series_description = None

    def __init__(self, file=None, processed_file=None, filter_name=None, study_uid=None, series_uid=None, series_description=None):
        self.file = file
        self.processed_file = processed_file
        self.filter_name = filter_name
        self.study_uid = study_uid
        self.series_uid = series_uid
        self.series_description = series_description

    def __str__(self):
        return (
            'file              : ' + str(self.study_uid) +os.linesep+
            'processed_file    : ' + str(self.study_uid) +os.linesep+
            'filter_name       : ' + str(self.study_uid) +os.linesep+
            'study_uid         : ' + str(self.study_uid) +os.linesep+
            'series_uid        : ' + str(self.series_uid) +os.linesep+
            'series_description: ' + str(self.series_uid))

    @staticmethod
    def from_string(string):
        fm = FileMetadata
        for line in string.splitlines():
            if line.startswith('file'):
                fm.file = line.split(':')[1].strip()
            elif line.startswith('processed_file'):
                fm.processed_file = line.split(':')[1].strip()
            elif line.startswith('filter_name'):
                fm.filter_name = line.split(':')[1].strip()
            elif line.startswith('study_uid'):
                fm.study_uid = line.split(':')[1].strip()
            elif line.startswith('series_uid'):
                fm.series_uid = line.split(':')[1].strip()
            if line.startswith('series_description'):
                fm.series_description = line.split(':')[1].strip()
        return fm

class Timeline:
    patient_id = None #ID of the Patient in PACS
    path = None #Path to the root timeline folder
    start_date = None #First date covered by timeline
    end_date = None #Last date covered by timeline
    brain_mask = None #Brain mask which will be used
    registration_reference = None #Reference scan for registration
    is_processed = False #Is the pre-processing up to date?
    datamap = {} #In-memory map of the data structure
    #^^ for now we'll try to use the file system to guide us
    filters = default_filters() #Types of scans to include (defaut FLAIR and MPRAGE)

    def __init__(self, path, patient_id=None):
        self.path = path
        self.patient_id = patient_id
        # If we don't have a patient ID we're assuming patient id as the folder name
        if patient_id == None:
            self.patient_id = os.path.basename(os.path.normpath(path))

        # Try to load from path...
        if (os.path.isfile(os.path.join(path,'timeline.metadata'))):
            self.load()

        # If that doesn't work, try to create from PACS
        if not os.path.exists(path): os.makedirs(path)
        self.save()

    def update_from_pacs(self, scp_settings):
        if scp_settings == None: return

        patient = Patient(self.patient_id, scp_settings)
        # Do we have new dates?
        for study in patient.find_studies():
            study_path = os.path.join(self.path, study.study_date)
            try:
                os.mkdir(study_path)
            except:
                pass

            # Create a new in-memory data map
            self.datamap[study.study_date] = []
            # Get filtered series
            for filter in self.filters:
                series = filter.filter(study)
                if series != None:
                    data = FileMetadata(file=filter.name + ".nii.gz")
                    new_series = True
                    metadatafile = os.path.join(study_path, data.file + '.metadata')
                    # Update existing metadata
                    if os.path.exists(metadatafile):
                        with open(metadatafile, 'r') as f:
                            try:
                                data.__dict__ = json.loads(f.read())
                            except:
                                raise Exception('Failed to read ' + metadatafile)
                        # If the series has changed, we'll delete the old one.
                        if data.series_uid != series.series_uid:
                            if os.path.exists(os.path.join(study_path, data.file)):
                                os.remove(os.path.join(study_path, data.file))
                            if os.path.exists(os.path.join(study_path, data.processed_file)):
                                os.remove(os.path.join(study_path, data.processed_file))
                        else:
                            new_series = False
                    # If we have a new (or replaced) series, update everything and get the data
                    if new_series:
                        data = FileMetadata(
                            file=filter.name + ".nii.gz",
                            processed_file=filter.name + ".processed.nii.gz",
                            filter_name=filter.name,
                            study_uid=series.study_uid,
                            series_uid=series.series_uid,
                            series_description=series.description)
                        self.datamap[study.study_date].append(data)
                        # Write metadata
                        with open(metadatafile, 'w') as f:
                            f.write(json.dumps(vars(data)))
                            f.flush()
                        series.save_nifti(os.path.join(study_path,data.file))

        self.is_processed = False
        self.save()

    def save(self):
        content = json.dumps(vars(self))
        with open(os.path.join(self.path, 'timeline.metadata'), 'w') as f:
            f.write(content)
        self._save_datamap()

    def load(self):
        with open(os.path.join(self.path, 'timeline.metadata'), 'r') as f:
            content = f.read()
            self.__dict__ = json.loads(content)
        self._load_datamap()

    def _save_datamap(self):
        for studydate in self.datamap:
            for filemeta in self.datamap[studydate]:
                # Create study directory if it's not there
                studypath = os.path.join(self.path, studydate)
                if not os.path.exists(studypath): os.makedirs(studypath)
                # Save metadata
                with open(os.path.join(studypath, filemeta.file + '.metadata'), 'w') as f:
                    f.write(json.dumps(vars(filemeta)))

    def _load_datamap(self):
        for studydate in next(os.walk(self.path))[1]:
            self.datamap[studydate] = []
            files = next(os.walk(os.path.join(self.path, studydate)))[2]
            files = [f for f in files if f.endswith('.metadata')]
            for f in files:
                with open(os.path.join(self.path, studydate, f), 'r') as f:
                    filemeta = FileMetadata()
                    try:
                        filemeta.__dict__ = json.loads(f.read())
                        self.datamap[studydate].append(filemeta)
                    except:
                        raise Exception('Failed to read ' + os.path.join(self.path, studydate, f))


    def setup_registration_reference(self):
        from pattools import ants
        print('Setting up registration reference...')
        # Check that we don't already have one
        if (self.registration_reference != None
            and os.path.exists(os.path.join(self.path,self.registration_reference))
            and self.brain_mask != None
            and os.path.exists(os.path.join(self.path,self.brain_mask))):
            return
        # For now we're just going to go with the biggest image as a proxy for high res
        largest_file = (None, -1)
        for dir, _, files in os.walk(self.path):
            for f in files:
                if (f.endswith('.nii') or f.endswith('.nii.gz')) and f+'.metadata' in files:
                    path = os.path.join(dir, f)
                    candidate = (path, os.stat(path).st_size)
                    if largest_file[0] == None or candidate[1] > largest_file[1]:
                        largest_file = candidate

        with TemporaryDirectory() as tmp_dir:
            #Open atlas
            atlas_path = '/data/atlas/mni'
            if not os.path.exists(atlas_path): os.makedirs(atlas_path)
            atlas = Atlas.MNI.load(atlas_path)

            # save atlas to tmp_dir and mask to timeline
            t2_path = os.path.join(tmp_dir, 't2.nii.gz')
            mask_path = os.path.join(tmp_dir, 'mask.nii.gz')

            nib.save(atlas.t2, t2_path)
            nib.save(atlas.mask, mask_path)

            # Bias correction and registration
            print('        N4 Bias correction for reference image...')
            n4_path = os.path.join(self.path, 'registration_reference.nii.gz')
            out_path = os.path.join(tmp_dir, 'waped.nii.gz')
            ants.n4_bias_correct(largest_file[0], n4_path).wait()

            print('        Registering reference image to brain mask...')
            ants.affine_registration(t2_path, n4_path, out_path).wait()
            # Register mask to reference scan
            affine_mat = out_path + '_0GenericAffine.mat'

            shutil.copyfile(affine_mat, os.path.join(self.path, 'atlas2ref.mat'))
            print(next(os.walk(tmp_dir)))
            out_path = os.path.join(self.path, 'brain_mask.nii.gz')
            ants.apply_linear_transform(mask_path, n4_path, affine_mat, out_path).wait()
            # Save metadata
            self.registration_reference = 'registration_reference.nii.gz'
            self.brain_mask = 'brain_mask.nii.gz'
            self.save()
            print('done.')

    def process_file(self, input_path, output_path, apply_mask=False):
        # These imports can complain on import, so we'll only get them now.
        from pattools import ants
        with TemporaryDirectory() as tmp_dir:
            input = nib.load(input_path)
            n4_path = os.path.join(tmp_dir, 'n4.nii')
            ants.n4_bias_correct(input_path, n4_path).wait()

            ref_path = os.path.join(self.path, self.registration_reference)
            out_path = os.path.join(tmp_dir, 'regout.nii')
            ants.affine_registration(n4_path, ref_path, out_path).wait()

            mask = nib.load(os.path.join(self.path, self.brain_mask))
            output = nib.load(out_path)
            if apply_mask:
                outdata = output.get_fdata() * mask.get_fdata()
            output = nib.Nifti1Image(outdata, output.affine, output.header)
            nib.save(output, output_path)

    def process(self):
        if (self.registration_reference == None
            or os.path.exists(os.path.join(self.path, self.registration_reference)) == False
            or self.brain_mask == None
            or os.path.exists(os.path.join(self.path, self.brain_mask)) == False):
            self.setup_registration_reference()

        files_to_process = []
        for study in self.datamap:
            study_path = os.path.join(self.path, study)
            for filter in self.datamap[study]:
                if not os.path.exists(os.path.join(filter.processed_file)):
                    input = os.path.join(self.path, study, filter.file)
                    output = os.path.join(self.path, study, filter.processed_file)
                    files_to_process.append((input, output))
        # Add a progress bar
        print('Processing', len(files_to_process), 'files...')
        files_to_process = progress.bar(files_to_process, expected_size=len(files_to_process))
        for input, output in files_to_process:
            self.process_file(input, output)
