'''Timelines help organise data longitudinally'''

from pattools.pacs import Series, Patient
from pattools.resources import Atlas
from pattools.image import histogram_match, normalize_by_whitematter, estimate_window
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
    is_rendered = False
    datamap = None #In-memory map of the data structure
    #^^ for now we'll try to use the file system to guide us
    manual_studies = None #This is a collection of studies which have been manually set to override the autmated matching.
    filters = None #Types of scans to include (defaut FLAIR and MPRAGE)
    normed_window_map = None

    def __init__(self, path, patient_id=None):
        self.path = os.path.abspath(path)
        self.patient_id = patient_id
        # If we don't have a patient ID we're assuming patient id as the folder name
        if patient_id == None:
            self.patient_id = os.path.basename(os.path.normpath(path))

        # Try to load from path...
        self.load()

        # If that doesn't work, try to create from PACS
        if not os.path.exists(path): os.makedirs(path, mode=0o777)

        # Hot tip: If we declare these at a class scope then __dict___ (and vars()) won't contain the entries.
        # Python will assume that the haven't changed because the reference hasn't changed.
        self.datamap = {}
        self.manual_studies = {}
        self.filters = pattools._timelinefilter.default_filters()
        self.normed_window_map = {}
        self.save()

    def clean(self):
        '''Removes any folders where the number of metadata files and image files don't match'''
        self.datamap = {}
        print('self.path',self.path)
        print(os.path.exists(self.path))
        if not os.path.exists(self.path): return

        to_clean = []
        for root, folders, files in os.walk(self.path):
            #Clean the directory if the stucture isn't well formed
            if (len(folders) == 0
                and len([f for f in files if f.endswith('.nii.gz') or f.endswith('.nii')])
                    < len([f for f in files if f.endswith('.metadata') and f != 'timeline.metadata'])):
                to_clean.append(os.path.join(self.path, root))
            # Or if there are no matching studies
            if len(folders) == 0 and len(files) == 0:
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
        if not os.path.exists(os.path.join(self.path, 'timeline.metadata')):
            print('No metadata file exists in ', os.path.join(self.path, 'timeline.metadata'))
            return

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
        print('Processing: ', input_path)
        from pattools import ants
        with TemporaryDirectory() as tmp_dir:
            n4_path = os.path.join(tmp_dir, 'n4.nii')
            ants.n4_bias_correct(input_path, n4_path).wait()

            ref_path = os.path.join(self.path, self.registration_reference)
            out_path = os.path.join(tmp_dir, 'regout.nii')
            ants.affine_registration(n4_path, ref_path, out_path).wait()

            mask = nib.load(os.path.join(self.path, self.brain_mask))
            output = nib.load(out_path)
            outdata = output.get_fdata() * 1 # maybe this will force into an ndarray?
            if apply_mask:
                outdata *= mask.get_fdata()

            output = nib.Nifti1Image(outdata, output.affine, output.header)
            nib.save(output, output_path)

    def normalize_file(self, file, reference_file):
         # normalise whitematter intensity
        if self.whitematter_mask != None and self.brain_mask != None:
            wm_mask = nib.load(os.path.join(self.path, self.whitematter_mask)).get_fdata()
            mask = nib.load(os.path.join(self.path, self.brain_mask)).get_fdata()
            img = nib.load(file)
            img_data = img.get_fdata()
            ref_data = nib.load(reference_file).get_fdata()
            outdata = normalize_by_whitematter(
                img_data * mask,
                ref_data * mask,
                wm_mask)

            output = nib.Nifti1Image(outdata, output.affine, output.header)
            nib.save(output, file)

    def normalize_data(self, filter_name, brain_mask=True, do_znorm=True, do_histogram_match=False):
        '''This will generate a slice-wise normalized set of data (which you'll need to fit in RAM)'''
        print('Normalizing data for filter', filter_name)

        mask = nib.load(os.path.join(self.path, self.brain_mask)).get_fdata() if brain_mask and self.brain_mask != None else 1.
        files = [
            os.path.join(self.path, fm.study_date, fm.processed_file)
            for fm in self.files_for_filter(filter_name)]

        if len(files) == 0:
            print('No files for filter', filter_name)
            return

        # Work out the dimensions for our output
        first_img = nib.load(files[0])
        first_data = first_img.get_fdata()

        # We're going to try to fit the whole thing into memory because we have tonnes of RAM
        # We could refactor this to work via HDD if that's an issue but you'd probably see thrashing.
        # The output ndarray will be coronal, study index (time), then the two other spatial dims (should check).
        output = np.zeros((first_data.shape[0], len(files),first_data.shape[1],first_data.shape[2]))

        # Load all the timeline data into a single array (better have the memory for it)
        for i, file in enumerate(files):
            data = nib.load(file).get_fdata()
            for slice_index in range(data.shape[0]):
                output[slice_index,i,:,:] = data[slice_index,:,:] * mask[slice_index,:,:]
        print(f'Loaded {len(files)} files in {output.size*output.itemsize / (1024.*1024.)}MB of RAM')

        # Background threshold for Z-Norm
        background_threshold = 20
        print('processing', output.shape[0], 'slices')
        for slice_index in range(output.shape[0]):
            print('       ', slice_index, '/', output.shape[0])
            # We're going to take the most recent study as our reference
            bg_mask = output[slice_index,-1,:,:] > background_threshold
            # Skip the background only slices.
            if (np.sum(bg_mask.astype(float)) <= 0):
                continue
            mean0 = np.mean(output[slice_index,-1,:,:][bg_mask])
            std0 = np.std(output[slice_index,-1,:,:][bg_mask])
            if mean0 == 0 or std0 == 0:
                continue
            for i in range(0, output.shape[1]): # Where i is the study index
                bg_mask = output[slice_index,i,:,:] > background_threshold
                if (np.sum(bg_mask.astype(float)) <= 0):
                    continue
                sliceI = output[slice_index,i,:,:]
                if len(sliceI) != 0:
                    meanI = np.mean(output[slice_index,i,:,:][bg_mask])
                    stdI = np.std(output[slice_index,i,:,:][bg_mask])
                    if meanI == 0 or stdI == 0:
                        continue
                    # Apply Z-Norm and Histogram Matching
                    if do_histogram_match:
                        output[slice_index,i,:,:] = histogram_match(output[slice_index,i,:,:], output[slice_index,0,:,:])
                    if do_znorm: 
                        output[slice_index,i,:,:] = (output[slice_index,i,:,:] - meanI) / stdI * std0 + mean0

        # Make sure the minimum is zero (so we don't get grey backgrounds)
        output = output - np.min(output)
        # Norm the data to full uint16 range (this makes export way easier)
        output = output / np.max(output) * np.iinfo(np.uint16).max
        #output = output.astype(np.uint16)
        # Export the data
        print('max, min:', np.max(output), ",", np.min(output))
        for i, f in enumerate(files):
            img = nib.load(f)
            nib.save(nib.Nifti1Image(output[:,i,:,:] * mask, img.affine, img.header), f)
        
        # TODO, this should probably be moved to the renderer
        wl, ww  = estimate_window(output) # Window level, and window width are the convention
        min_val = np.min(output)
        max_val = np.max(output)
        if min_val == np.nan: min_val = 0
        if max_val == np.nan: max_val = 0
        self.normed_window_map[filter_name] = {'min':min_val, 'max':max_val, 'ww': ww, 'wl': wl}
        for k in self.normed_window_map[filter_name].keys():
            if self.normed_window_map[filter_name][k] == np.NaN or self.normed_window_map[filter_name][k] == np.Infinity:
                self.normed_window_map[filter_name][k] = 0
        print(self.normed_window_map)
        self.save()

    def process(self):
        '''Process (bias correct, register to reference, etc.) all image files'''
        if (self.registration_reference == None
            or os.path.exists(os.path.join(self.path, self.registration_reference)) == False
            or self.brain_mask == None
            or os.path.exists(os.path.join(self.path, self.brain_mask)) == False):
            self.setup_registration_reference()

        files_to_process = []
        self._load_datamap()
        # TODO -- Not currently using the given histogram reference, need to either remove or use...
        histogram_references = {}
        # TODO -- We seem to be selecting the most recent study as the histogram match
        # this will break on update
        #for filter in self.filters:
            #histogram_ref_path = os.path.join(self.path, f'intensity_ref_{filter.name}.nii.gz')
            #if not os.path.exists(histogram_ref_path):
            #    candidate_path = ""
            #    for study_date in self.datamap:
            #        for fm in self.datamap[study_date]:
            #            if fm.filter_name == filter.name:
            #                candidate_path = os.path.join(self.path, study_date, fm.processed_file)
            #                if not os.path.exists(candidate_path):
            #                    candidate_path = os.path.join(self.path, study_date, fm.file)
            #    try:
            #        shutil.copyfile(candidate_path, histogram_ref_path)
            #    except:
            #        print('Failed to copy histogram reference ', candidate_path)
            #histogram_references[filter.name] = histogram_ref_path

        for study in self.datamap:
            study_path = os.path.join(self.path, study)
            for filemeta in self.datamap[study]:
                print(os.path.join(study_path, filemeta.processed_file), 'exists?', os.path.exists(os.path.join(study_path, filemeta.processed_file)))
                if os.path.exists(os.path.join(study_path, filemeta.processed_file)) == False and os.path.exists(os.path.join(study_path, filemeta.file)) == True:
                    input = os.path.join(self.path, study, filemeta.file)
                    output = os.path.join(self.path, study, filemeta.processed_file)
                    filter_name = filemeta.filter_name
                    files_to_process.append((input, output))
        # Add a progress bar
        print('Processing', len(files_to_process), 'files...')
        files_to_process = progress.bar(files_to_process, expected_size=len(files_to_process))
        for input, output in files_to_process: # TODO: Discarding histogram_reference
            self.process_file(input, output)

        # Now that all the files are processed (including the histogram references) we can normalize
        for fil in self.filters:
            self.normalize_data(fil.name)

    def study_dates(self):
        '''Returns all study dates, whether we have a scan or not'''
        folders = next(os.walk(self.path))[1]
        return [datetime.strptime(f, '%Y%m%d').date() for f in folders]

    def set_manual_series(self, filter_name, study_date, series):
        '''Sets a manually selected series for a given filter/study combination'''
        self.manual_studies[filter_name, study_date] = series

    def files_for_filter(self, filter_name):
        '''Returns the file metadata objects for a given filtername, sorted by date'''
        result = []
        self.load() # Load datamap from disk
        for study in self.datamap.keys():   
            for filemeta in self.datamap[study]:
                if filemeta.filter_name == filter_name:
                    result.append(filemeta)

        result.sort(key=lambda fm: fm.study_date)
        return result
