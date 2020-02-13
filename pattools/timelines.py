'''Timelines help organise data longitudinally'''

from pattools.pacs import Series, Patient
from pattools.resources import Atlas
import nibabel as nib
import numpy as np
import multiprocessing
from joblib import Parallel, delayed
import json
import os
import shutil
from clint.textui import progress
from tempfile import TemporaryDirectory
from datetime import date, timedelta, datetime
import imageio

class ScorecardElement:
    '''The scorecard element is used to create a series filter.'''
    description = None
    points = 0
    def __init__(self, description, points):
        self.description = description
        self.points = points

class Filter:
    '''The filter class extracts a 0 or 1 series from a study (which math the filter, obvs.)'''
    name = None
    scorecard = None # For matching description
    min_rows = 100
    min_cols = 100
    threshold = 0

    def __init__(self, name):
        self.name = name
        self.scorecard = []

    def filter(self, study):
        '''Returns the series which best matches the filter, or None if there are no good matches.'''
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
    '''Default filter for FLAIR studies'''
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
    '''Default filter for MPRAGE studies'''
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
    '''Default filter list'''
    return [flair_filter(), mprage_filter()]

class FileMetadata:
    '''Metadata for a stored image file'''
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
            'processed_file    : ' + str(self.processed_file) +os.linesep+
            'filter_name       : ' + str(self.filter_name) +os.linesep+
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
    brain_mask = None #Brain mask which will be used
    registration_reference = None #Reference scan for registration
    is_processed = False #Is the pre-processing up to date?
    is_rendererd = False
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
        '''Populate the Timeline from PACS'''
        if scp_settings == None: return

        patient = Patient(self.patient_id, scp_settings)
        self.patient_name = patient.name
        self.patient_dob = patient.dob
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
                        self.datamap[study.study_date].append(data)
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
        '''Save to disk'''
        content = json.dumps(vars(self))
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
        self._load_datamap()

    def _save_datamap(self):
        '''Save just the datamap metadata'''
        for studydate in self.datamap:
            for filemeta in self.datamap[studydate]:
                # Create study directory if it's not there
                studypath = os.path.join(self.path, studydate)
                if not os.path.exists(studypath): os.makedirs(studypath)
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
        # For now we're just going to go with the biggest image as a proxy for high res
        largest_file = (None, -1)
        candidates = []
        for dir, _, files in os.walk(self.path):
            for f in files:
                if (f.endswith('.nii') or f.endswith('.nii.gz')) and f+'.metadata' in files:
                    candidates.append(os.path.join(dir, f))
        # candidates are sorted by the minimum dimension size
        candidates.sort(key=lambda c : min(nib.load(c).shape))
        if len(candidates) == 0: return
        # return the candidate with the largest minimum dimension size
        candidate = candidates[-1]
        print('candidate:', candidate)

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
            out_path = os.path.join(tmp_dir, 'warped.nii.gz')
            ants.n4_bias_correct(candidate, n4_path).wait()

            print('        Registering brain mask to reference image...')
            # Register mask to reference scan
            ants.syn_registration(t2_path, n4_path, out_path).wait()
            # These will be the output of the registration
            affine_mat = out_path + '_0GenericAffine.mat'
            inverse_warp = out_path + '_1InverseWarp.nii.gz'
            warp = out_path + '_1Warp.nii.gz'
            # Keep them handy
            shutil.copyfile(affine_mat, os.path.join(self.path, 'affine_from_MNI.mat'))
            shutil.copyfile(inverse_warp, os.path.join(self.path, 'warp_to_MNI.nii.gz'))
            shutil.copyfile(warp, os.path.join(self.path, 'warp_from_MNI.nii.gz'))
            # Apply affine transform then warp to put mask in registered space
            out_path = os.path.join(self.path, 'brain_mask.nii.gz')
            ants.apply_transform(mask_path, n4_path, affine_mat, out_path, warp=warp).wait()
            # Save metadata
            self.registration_reference = 'registration_reference.nii.gz'
            self.brain_mask = 'brain_mask.nii.gz'
            self.save()
            print('done.')

    def process_file(self, input_path, output_path, apply_mask=False):
        '''Process (biascorrect, register, etc.) a single file'''
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
            outdata = output.get_fdata()
            if apply_mask:
                outdata *= mask.get_fdata()
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
        for study in self.datamap:
            study_path = os.path.join(self.path, study)
            for filemeta in self.datamap[study]:
                print(os.path.exists(os.path.join(study_path, filemeta.processed_file)))
                if not os.path.exists(os.path.join(study_path, filemeta.processed_file)):
                    input = os.path.join(self.path, study, filemeta.file)
                    output = os.path.join(self.path, study, filemeta.processed_file)
                    files_to_process.append((input, output))
        # Add a progress bar
        print('Processing', len(files_to_process), 'files...')
        files_to_process = progress.bar(files_to_process, expected_size=len(files_to_process))
        for input, output in files_to_process:
            self.process_file(input, output)

    def study_dates(self):
        '''Returns all study dates, whether we have a scan or not'''
        folders = next(os.walk(self.path))[1]
        return [datetime.strptime(f, '%Y%m%d').date() for f in folders]


 #############################
######## INTERPOLATORS ########
 #############################

class _AbstractInterpolator:
    '''Base abstration for interpolators'''
    def interpolate(self, data1, data2, ratio):
        raise Exception("This is the base interpolator class. Use an implementation")

    @staticmethod
    def _date_from_path(path):
        datestring = os.path.basename(os.path.dirname(path))
        return date(int(datestring[0:4]), int(datestring[4:6]), int(datestring[6:]))

    def interpolated_data(self, image_paths, mask_path, delta_days):
        ''' Returns a list of numpy volumes interpolated based on the delta days. All real scans are included.'''
        # We only want to yield data2 on the final pair, so we'll need a reference
        data2 = None
        date2 = None
        for p1, p2 in zip(image_paths[:-1], image_paths[1:]):
            # Do some error checking...
            if not os.path.exists(p1):
                raise Exception('Path ' + p1 + ' does not exist')
            if not os.path.exists(p2):
                raise Exception('Path ' + p2 + ' does not exist')
            if mask_path != None and os.path.exists(mask_path) == False:
                raise Exception('Path ' + mask_path + ' does not exist')

            # Open the nifti file
            p1img = nib.load(p1)
            p2img = nib.load(p2)
            # Get the data
            data1 = p1img.get_fdata()
            data2 = p2img.get_fdata()

            if (mask_path != None):
                mask_img = nib.load(mask_path)
                mask_data = mask_img.get_fdata()
                data1 *= mask_data
                data2 *= mask_data

            # If each delta represents a step, we calculate how many steps there
            # are between the current scans
            date1 = _AbstractInterpolator._date_from_path(p1)
            date2 = _AbstractInterpolator._date_from_path(p2)
            window = (date2 - date1)
            steps = int(window.days / delta_days)

            for i in range(0, steps):
                ratio = i/steps
                # Yield interpolated (includes data1, since the ratio range is [0,1)
                yield (date1 + timedelta(days=int(i * delta_days)), self.interpolate(data1, data2, ratio))
        # Yield the last series
        yield (date2, data2)

class LinearInterpolator(_AbstractInterpolator):
    '''Interpolates data linearly'''
    def __init__(self):
        super().__init__()

    def interpolate(self, data1, data2, ratio):
        return data1 * (1-ratio) + data2 * ratio

class NearestNeighbourInterpolator(_AbstractInterpolator):
    '''Returns the nearest real scan'''
    def __init__(self):
        super().__init__()

    def interpolate(self, data1, data2, ratio):
        if ratio >= 0.5: return data2
        return data1

class NullInterpolator(_AbstractInterpolator):
    '''The null iterpolator returns only the masked data (with no interpolation)'''
    def __init__(self):
        super().__init__()

    def interpolate(self, data1, data2, ratio):
        pass

    def interpolated_data(self, image_paths, mask_path, delta_days=None):
        for path in image_paths:
            # Open the nifti file
            img = nib.load(path)
            # Get the data
            data = img.get_fdata()

            if (mask_path != None):
                mask_img = nib.load(mask_path)
                mask_data = mask_img.get_fdata()
                data *= mask_data

            date = _AbstractInterpolator._date_from_path(path)
            yield (date, data)

 #########################
######## Renderers ########
 #########################

class Renderer:
    '''Renders interpolated data to image files.'''
    interpolator = None
    days_delta = None

    def __init__(self, interpolator=LinearInterpolator(), days_delta=28):
        self.interpolator = interpolator
        self.days_delta = days_delta

    @staticmethod
    def _get_files(timeline, filter):
        files = []
        for studydate in timeline.datamap:
            files.extend([
                os.path.join(timeline.path, studydate, fm.processed_file)
                for fm in timeline.datamap[studydate]
                if fm.filter_name == filter.name])
        return files

    def render(self, timeline, path):
        '''Write images to path given based on a timeline. Files will be interpolated and rendered to <path>/<filter>/<cor|sag|ax>/<date>/'''
        for filter in timeline.filters:
            files = Renderer._get_files(timeline, filter)
            self.render_all(files, os.path.join(timeline.path, timeline.brain_mask), os.path.join(path, filter.name))

    def render_new_studies(self, timeline, path):
        '''Write images to path given based on a timeline. Files will be interpolated and rendered to <path>/<filter>/<cor|sag|ax>/<date>/'''
        for filter in timeline.filters:
            files = Renderer._get_files(timeline, filter)
            self.render_new(files, os.path.join(timeline.path, timeline.brain_mask), os.path.join(path, filter.name))

    @staticmethod
    def write_images(data, folder, slice_type, min_val, max_val):
        if not os.path.exists(folder):
            os.makedirs(folder)
        data_cp = np.copy(data)
        count = 0
        if slice_type == 'sag':
            count = data.shape[0]
            for i in range(data.shape[0]):
                Renderer.write_image(data_cp[i,:,:], os.path.join(folder, f'{i}.png'), min_val, max_val)

        elif slice_type == 'cor':
            count = data.shape[1]
            for j in range(data.shape[1]):
                Renderer.write_image(data_cp[:,j,:], os.path.join(folder, f'{j}.png'), min_val, max_val)

        elif slice_type == 'ax':
            count = data.shape[2]
            for k in range(data.shape[2]):
                Renderer.write_image(data_cp[:,:,k], os.path.join(folder, f'{k}.png'), min_val, max_val)

        return count

    @staticmethod
    def write_image(slice, location, min, max):
        '''Write a single slice to an image at the given location'''
        # This is a bit of a hack to make sure the range is normal
        slice[0,0] = max
        slice[0,1] = min
        output = np.flip(slice.T).copy()
        np.clip(output, min, max)
        imageio.imwrite(location, output)

    @staticmethod
    def _render_volume(date, volume, path, overwrite=True):
        '''Render every slice in a volume along 3 axis.'''
        min_val = np.amin(volume)
        max_val = np.amax(volume)
        # Output paths
        sag_path = os.path.join(path, date.strftime('%Y%m%d'), 'sag')
        cor_path = os.path.join(path, date.strftime('%Y%m%d'), 'cor')
        ax_path = os.path.join(path, date.strftime('%Y%m%d'), 'ax')
        # Write images if the folder doesn't exist or overwrite is true
        if overwrite or os.path.exists(sag_path) == False:
            Renderer.write_images(volume, sag_path, 'sag', min_val, max_val)
        if overwrite or os.path.exists(cor_path) == False:
            Renderer.write_images(volume, cor_path, 'cor', min_val, max_val)
        if overwrite or os.path.exists(ax_path) == False:
            Renderer.write_images(volume, ax_path, 'ax', min_val, max_val)

    def render_all(self, files, mask_path, path):
        '''Render all volumes using supplied brain mask'''
        Parallel(n_jobs=multiprocessing.cpu_count())(
            delayed(Renderer._render_volume)(date, volume, path, overwrite=True)
            for date, volume in self.interpolator.interpolated_data(files, mask_path, self.days_delta))

    def render_new(self, files, mask_path, path):
        '''Render new volumes using supplied brain mask'''
        Parallel(n_jobs=multiprocessing.cpu_count())(
            delayed(Renderer._render_volume)(date, volume, path, overwrite=False)
            for date, volume in self.interpolator.interpolated_data(files, mask_path, self.days_delta))
