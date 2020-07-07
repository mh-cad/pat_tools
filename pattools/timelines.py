'''Timelines help organise data longitudinally'''

from pattools.pacs import Series, Patient
from pattools.resources import Atlas
from pattools.image import histogram_match, normalize_by_whitematter
import nibabel as nib
import numpy as np
import json
import os
import shutil
from clint.textui import progress
from tempfile import TemporaryDirectory
from datetime import date, timedelta, datetime

#local imports
import pattools._timelinefilter

class FileMetadata:
    '''Metadata for a stored image file'''
    file = None
    processed_file = None
    filter_name = None
    study_date = None
    study_uid = None
    series_uid = None
    series_description = None

    def __init__(self, file=None, processed_file=None, filter_name=None, study_date=None, study_uid=None, series_uid=None, series_description=None):
        self.file = file
        self.processed_file = processed_file
        self.filter_name = filter_name
        self.study_date = study_date
        self.study_uid = study_uid
        self.series_uid = series_uid
        self.series_description = series_description

    def __str__(self):
        return (
            'file              : ' + str(self.study_uid) +os.linesep+
            'processed_file    : ' + str(self.processed_file) +os.linesep+
            'filter_name       : ' + str(self.filter_name) +os.linesep+
            'study_date       : ' + str(self.study_date) +os.linesep+
            'study_uid         : ' + str(self.study_uid) +os.linesep+
            'series_uid        : ' + str(self.series_uid) +os.linesep+
            'series_description: ' + str(self.series_description))

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
            elif line.startswith('study_date'):
                fm.study_date = line.split(':')[1].strip()
            if line.startswith('series_description'):
                fm.series_description = line.split(':')[1].strip()
        return fm

class Timeline:
    '''The timeline contains filtered lists of scans over a single patient's history'''
    patient_id = None #ID of the Patient in PACS
    patient_name = None
    patient_dob = None
    path = None #Path to the root timeline folder
    start_date = None #First date covered by timeline
    end_date = None #Last date covered by timeline
    brain_mask = None #Brain mask which will be used\
    whitematter_mask = None #Whitematter mask
    registration_reference = None #Reference scan for registration
    is_processed = False #Is the pre-processing up to date?
    is_rendererd = False
    datamap = None #In-memory map of the data structure
    #^^ for now we'll try to use the file system to guide us
    manual_studies = None #This is a collection of studies which have been manually set to override the autmated matching.
    filters = None #Types of scans to include (defaut FLAIR and MPRAGE)

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
        if not os.path.exists(path): os.makedirs(path, mode=0o777)

        # Hot tip: If we declare these at a class scope then __dict___ (and vars()) won't contain the entries.
        # Python will assume that the haven't changed because the reference hasn't changed.
        self.datamap = {}
        self.manual_studies = {}
        self.filters = pattools._timelinefilter.default_filters()

        self.save()

    def clean(self):
        '''Removes any folders where the number of metadata files and image files don't match'''
        self.datamap = {}

        if not os.path.exists(self.path): return

        to_clean = []
        for root, folders, files in os.walk(self.path):
            if (len(folders) == 0
                and len([f for f in files if f.endswith('.nii.gz') or f.endswith('.nii')])
                    < len([f for f in files if f.endswith('.metadata') and f != 'timeline.metadata'])):
                to_clean.append(os.path.join(self.path, root))

        print('cleaning:', to_clean)
        for root in to_clean:
            shutil.rmtree(root)

    def update_from_pacs(self, scp_settings):
        '''Populate the Timeline from PACS'''
        print('Updating from PACS...')

        if scp_settings == None:
            print('No SCP Settings found.')
            return

        patient = Patient(self.patient_id, scp_settings)
        self.patient_name = patient.name
        self.patient_dob = patient.dob
        # Do we have new dates?

        # Sometimes there will be studies which showed up and created metadata,
        # but don't have an image. Then they dissapear and ruin everything.
        # TODO: Find out why.
        # For now we'll just clean directories where the metadata count != the image count.
        self.clean()
        try:
            print('finding studies...')
            for study in patient.find_studies():
                study_path = os.path.join(self.path, study.study_date)
                print('study path:', study_path)
                try:
                    os.mkdir(study_path)
                except:
                    pass

                # Create a new in-memory data map
                self.datamap[study.study_date] = []
                print('study date:', study.study_date)
                # Get filtered series
                for filter in self.filters:
                    series = None
                    # check for a manually selected series for the study / filter combo
                    if (filter.name, study.study_date) in self.manual_studies:
                        series = self.manual_studies[filter.name, study.study_date]
                    else:
                        series = filter.filter(study)

                    if series != None:
                        print('series:', series.description)
                        data = FileMetadata(file=filter.name + ".nii.gz")
                        new_series = True
                        metadatafile = os.path.join(study_path, data.file + '.metadata')
                        # Update existing metadata
                        if os.path.exists(metadatafile):
                            print('metadata found')
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
                                # There has been no change and the file exists.
                                if os.path.exists(os.path.join(study_path, data.file)):
                                    print('nifti found and has not changed')
                                    new_series = False

                            self.datamap[study.study_date].append(data)
                        # If we have a new (or replaced) series, update everything and get the data
                        if new_series:
                            data = FileMetadata(
                                file=filter.name + ".nii.gz",
                                processed_file=filter.name + ".processed.nii.gz",
                                study_date=study.study_date,
                                filter_name=filter.name,
                                study_uid=series.study_uid,
                                series_uid=series.series_uid,
                                series_description=series.description)
                            self.datamap[study.study_date].append(data)
                            # Write metadata
                            with open(metadatafile, 'w') as f:
                                f.write(json.dumps(vars(data)))
                                f.flush()
                            print('downloading:',os.path.join(study_path,data.file))
                            series.save_nifti(os.path.join(study_path,data.file))
                            if os.path.exists(os.path.join(study_path, data.file)):
                                print('success')
                            else:
                                print('failed')

                        # Try to re-download original file if we don't have it
                        if not os.path.exists(os.path.join(study_path, data.file)):
                            print("Can't find the series, let's try again...")
                            series.save_nifti(os.path.join(study_path,data.file))

            self.is_processed = False
            self.save()
        except Exception as e:
            print('Error occurred while updating from PACS', e)

    def save(self):
        '''Save to disk'''
        content = json.dumps(vars(self), default=lambda o: o.__dict__, sort_keys=True, indent=4)
        with open(os.path.join(self.path, 'timeline.metadata'), 'w') as f:
            f.write(content)
        self._save_datamap()

    def load(self):
        '''Load from disk'''
        with open(os.path.join(self.path, 'timeline.metadata'), 'r') as f:
            content = f.read()
            path = self.path
            self.__dict__ = json.loads(content)
            self.path = path # The saved path may be different but this is where we loaded it.
            #These filters will load as dicts but we want them to not be. So we need to parse 'em.
            parsed_filters = []
            for filter_dict in self.__dict__['filters']:
                f = pattools._timelinefilter.Filter(filter_dict['name'])
                for key in filter_dict:
                    setattr(f,key, filter_dict[key])
                parsed_filters.append(f)
            self.filters = parsed_filters
        self._load_datamap()

    def _save_datamap(self):
        '''Save just the datamap metadata'''
        for studydate in self.datamap:
            for filemeta in self.datamap[studydate]:
                # Create study directory if it's not there
                studypath = os.path.join(self.path, studydate)
                if not os.path.exists(studypath): os.makedirs(studypath, mode=0o777)
                # Save metadata
                with open(os.path.join(studypath, filemeta.file + '.metadata'), 'w') as f:
                    f.write(json.dumps(vars(filemeta)))

    def _load_datamap(self):
        '''Load just the datamap metadata'''
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
        '''Select a registration reference from the image data'''
        from pattools import ants
        print('Setting up registration reference...')
        # Check that we don't already have one
        if (self.registration_reference != None
            and os.path.exists(os.path.join(self.path,self.registration_reference))
            and self.brain_mask != None
            and os.path.exists(os.path.join(self.path,self.brain_mask))):
            return
        # Probably better ways to pick a candidate but we're going with "large but not too large"
        candidates = []
        for dir, _, files in os.walk(self.path):
            for f in files:
                if (f.endswith('.nii') or f.endswith('.nii.gz')) and f+'.metadata' in files:
                    candidates.append(os.path.join(dir, f))
        # candidates are sorted by the minimum dimension size
        candidates.sort(key=lambda c : min(nib.load(c).shape)) # Sort by minimum slices
        #candidates.sort(key=lambda c : os.stat(c).st_size) # Sort by file size
        if len(candidates) == 0: return
        # return the candidate with the largest minimum dimension size
        candidate = candidates[int(len(candidates) * 3 / 4)]
        print('candidate:', candidate)

        with TemporaryDirectory() as tmp_dir:
            #Open atlas
            atlas_path = os.path.join(self.path, '../atlas/mni')
            if not os.path.exists(atlas_path): os.makedirs(atlas_path, mode=0o777)
            atlas = Atlas.MNI.load(atlas_path)

            # save atlas to tmp_dir and mask to timeline
            t2_path = os.path.join(tmp_dir, 't2.nii.gz')
            mask_path = os.path.join(tmp_dir, 'mask.nii.gz')
            whitematter_mask_path = os.path.join(tmp_dir, 'whitematter_mask.nii.gz')

            nib.save(atlas.t2, t2_path)
            nib.save(atlas.mask, mask_path)
            nib.save(atlas.whitematter_mask, whitematter_mask_path)

            # Bias correction and registration
            print('        N4 Bias correction for reference image...')
            n4_path = os.path.join(self.path, 'registration_reference.nii.gz')
            out_path = os.path.join(tmp_dir, 'warped.nii.gz')
            ants.n4_bias_correct(candidate, n4_path).wait()

            print('        Registering brain mask to reference image...')
            # Register mask to reference scan
            ants.affine_registration(t2_path, n4_path, out_path).wait()
            # These will be the output of the registration
            affine_mat = out_path + '_0GenericAffine.mat'
            #inverse_warp = out_path + '_1InverseWarp.nii.gz'
            #warp = out_path + '_1Warp.nii.gz'
            # Keep them handy
            shutil.copyfile(affine_mat, os.path.join(self.path, 'affine_from_MNI.mat'))
            #shutil.copyfile(inverse_warp, os.path.join(self.path, 'warp_to_MNI.nii.gz'))
            #shutil.copyfile(warp, os.path.join(self.path, 'warp_from_MNI.nii.gz'))
            # Apply affine transform then warp to put mask in registered space
            out_path = os.path.join(self.path, 'brain_mask.nii.gz')
            ants.apply_transform(mask_path, n4_path, affine_mat, out_path).wait()
            white_out_path = os.path.join(self.path, 'whitematter_mask.nii.gz')
            ants.apply_transform(whitematter_mask_path, n4_path, affine_mat, white_out_path).wait()

            # Save metadata
            self.registration_reference = 'registration_reference.nii.gz'
            self.brain_mask = 'brain_mask.nii.gz'
            self.whitematter_mask = 'whitematter_mask.nii.gz'
            self.save()
            print('done.')

    def process_file(self, input_path, output_path, histogram_reference=None, apply_mask=False):
        '''Process (biascorrect, register, etc.) a single file'''
        # These imports can complain on import, so we'll only get them now.
        from pattools import ants
        with TemporaryDirectory() as tmp_dir:
            n4_path = os.path.join(tmp_dir, 'n4.nii')
            ants.n4_bias_correct(input_path, n4_path).wait()

            ref_path = os.path.join(self.path, self.registration_reference)
            out_path = os.path.join(tmp_dir, 'regout.nii')
            ants.affine_registration(n4_path, ref_path, out_path).wait()

            mask = nib.load(os.path.join(self.path, self.brain_mask))
            hist_ref = nib.load(os.path.join(self.path, histogram_reference))
            output = nib.load(out_path)
            outdata = output.get_fdata() * 1 # maybe this will force into an ndarray?
            if apply_mask:
                outdata *= mask.get_fdata()
            # normalise whitematter intensity
            if self.whitematter_mask != None and self.brain_mask != None and histogram_reference != None:
                whitematter_path = os.path.join(self.path, self.whitematter_mask)
                outdata = normalize_by_whitematter(
                    outdata * mask.get_fdata(),
                    hist_ref.get_fdata() * mask.get_fdata(),
                    nib.load(whitematter_path).get_fdata())

            output = nib.Nifti1Image(outdata, output.affine, output.header)
            nib.save(output, output_path)

    def process(self):
        '''Process (bias correct, register to reference, etc.) all image files'''
        if (self.registration_reference == None
            or os.path.exists(os.path.join(self.path, self.registration_reference)) == False
            or self.brain_mask == None
            or os.path.exists(os.path.join(self.path, self.brain_mask)) == False):
            self.setup_registration_reference()

        files_to_process = []
        self._load_datamap()
        histogram_references = {}
        for fm in self.datamap[list(self.datamap)[-1]]:
              histogram_references[fm.filter_name] = os.path.join(self.path, fm.study_date, fm.file)

        for study in self.datamap:
            study_path = os.path.join(self.path, study)
            for filemeta in self.datamap[study]:
                print(os.path.exists(os.path.join(study_path, filemeta.processed_file)))
                if not os.path.exists(os.path.join(study_path, filemeta.processed_file)):
                    input = os.path.join(self.path, study, filemeta.file)
                    output = os.path.join(self.path, study, filemeta.processed_file)
                    filter_name = filemeta.filter_name
                    files_to_process.append((input, output, histogram_references[filter_name]))
        # Add a progress bar
        print('Processing', len(files_to_process), 'files...')
        files_to_process = progress.bar(files_to_process, expected_size=len(files_to_process))
        for input, output, histogram_reference in files_to_process:
            self.process_file(input, output, histogram_reference=histogram_reference)

    def study_dates(self):
        '''Returns all study dates, whether we have a scan or not'''
        folders = next(os.walk(self.path))[1]
        return [datetime.strptime(f, '%Y%m%d').date() for f in folders]

    def set_manual_series(self, filter_name, study_date, series):
        '''Sets a manually selected series for a given filter/study combination'''
        self.manual_studies[filter_name, study_date] = series

