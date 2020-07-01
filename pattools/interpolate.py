import os
from datetime import date, timedelta, datetime
import nibabel as nib
import numpy as np
import imageio
from joblib import Parallel, delayed
import multiprocessing

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

    @staticmethod
    def interpolated_dates(dates, delta_days):
        ''' Returns a set of interpolated dates from a given list of dates and delta in days.'''
        result_dates = []
        for date1, date2 in zip(dates[:-1], dates[1:]):
            result_dates.append(date1)
            window = date2 - date1
            steps = int(window.days / delta_days)
            for i in range(0, steps):
                result_dates.append(date1+timedelta(days=int(i*delta_days)))
        # Add the last date
        result_dates.append(dates[-1])
        print('result_dates:', result_dates)
        return result_dates


    def _data_for_date(self, date, study_dates, study_paths, mask_data):
        ''' This method will return the interpolated data for a given date.
            We are assuming that the study_dates and study_paths are in the same order, which is sorted by date '''
        if date in study_dates:
            data = nib.load(study_paths[study_dates.index(date)]).get_fdata() * mask_data
            return (date, data)
        else:
            # Get the closest two study dates
            np_study_dates = np.unique(np.asarray(study_dates))
            closest_dates = np.argsort(np.abs(np_study_dates - date))[0:2]
            idx_before = min(closest_dates)
            idx_after = max(closest_dates)
            # Load the data from those dates
            before_data = nib.load(study_paths[idx_before]).get_fdata() * mask_data
            after_data = nib.load(study_paths[idx_after]).get_fdata() * mask_data
            # Work out the ratio for interpolation
            before_date = study_dates[idx_before]
            after_date = study_dates[idx_after]
            after_delta = after_date - before_date
            delta = date - before_date
            ratio = 0.
            if after_delta != 0:
                print('after_delta', after_delta)
                ratio = delta / after_delta
            # yield the interpolated result
            return (date, self.interpolate(before_data, after_data, ratio))

    def interpolated_data_from_dates(self, image_paths, mask_path, dates):
        study_dates = [_AbstractInterpolator._date_from_path(p) for p in image_paths]
        mask_data = None
        if mask_path != None and os.path.exists(mask_path):
            mask_data = nib.load(mask_path).get_fdata()
        else:
            print('mask path ' + str(mask_path) + ' does not exist')
            raise Exception('mask path ' + str(mask_path) + ' does not exist')

        for date in dates:
            d, d2 = self._data_for_date(date, study_dates, image_paths, mask_data)
            yield d, d2

    def interpolated_data_from_delta(self, image_paths, mask_path, delta_days):
        ''' Returns a list of numpy volumes interpolated based on the delta days. All real scans are included.'''
        all_dates = _AbstractInterpolator.interpolated_dates([_AbstractInterpolator._date_from_path(p) for p in image_paths], delta_days)
        return self.interpolated_data_from_dates(image_paths, mask_path, all_dates)
        # We only want to yield data2 on the final pair, so we'll need a reference
        #data2 = None
        #date2 = None
        ## Handle the empty case
        #if len(image_paths) == 0: return
        ## Handle the case of 1 image
        #if len(image_paths) == 1:
        #    if mask_path != None and os.path.exists(mask_path):
        #        yield (_AbstractInterpolator._date_from_path(image_paths[0]), nib.load(image_paths[0]).get_fdata() * nib.load(mask_path).get_fdata())
        #    else:
        #        raise Exception('Path ' + mask_path + ' does not exist')
        #    return
        ## Handle 2 or more images
        #for p1, p2 in zip(image_paths[:-1], image_paths[1:]):
        #    # Do some error checking...
        #    if not os.path.exists(p1):
        #        raise Exception('Path ' + p1 + ' does not exist')
        #    if not os.path.exists(p2):
        #        raise Exception('Path ' + p2 + ' does not exist')
        #    if mask_path != None and os.path.exists(mask_path) == False:
        #        raise Exception('Path ' + mask_path + ' does not exist')

        #    # Open the nifti file
        #    p1img = nib.load(p1)
        #    p2img = nib.load(p2)
        #    # Get the data
        #    data1 = p1img.get_fdata()
        #    data2 = p2img.get_fdata()

        #    if (mask_path != None):
        #        mask_img = nib.load(mask_path)
        #        mask_data = mask_img.get_fdata()
        #        data1 *= mask_data
        #        data2 *= mask_data

        #    # If each delta represents a step, we calculate how many steps there
        #    # are between the current scans
        #    date1 = _AbstractInterpolator._date_from_path(p1)
        #    date2 = _AbstractInterpolator._date_from_path(p2)
        #    window = (date2 - date1)
        #    steps = int(window.days / delta_days)

        #    for i in range(0, steps):
        #        ratio = i/steps
        #        # Yield interpolated (includes data1, since the ratio range is [0,1)
        #        yield (date1 + timedelta(days=int(i * delta_days)), self.interpolate(data1, data2, ratio))
        ## Yield the last series
        #yield (date2, data2)


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
            yield date, data

 #########################
######## Renderers ########
 #########################

class Renderer:
    '''Renders interpolated data to image files.'''
    interpolator = None
    days_delta = None
    timeline = None

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
        mask_path = None
        if timeline.path != None and timeline.brain_mask != None:
             mask_path = os.path.join(timeline.path, timeline.brain_mask)

        for filter in timeline.filters:
            files = Renderer._get_files(timeline, filter)
            self.render_all(files, mask_path, os.path.join(path, filter.name), timeline.study_dates())

    def render_new_studies(self, timeline, path):
        '''Write images to path given based on a timeline. Files will be interpolated and rendered to <path>/<filter>/<cor|sag|ax>/<date>/'''
        mask_path = None
        if timeline.path != None and timeline.brain_mask != None:
             mask_path = os.path.join(timeline.path, timeline.brain_mask)
        for filter in timeline.filters:
            print('filter:', filter)
            files = Renderer._get_files(timeline, filter)
            self.render_new(files, mask_path, os.path.join(path, filter.name), timeline.study_dates())

    @staticmethod
    def write_images(data, folder, slice_type, min_val, max_val):
        if not os.path.exists(folder):
            os.makedirs(folder, mode=0o777)
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

    def render_all(self, files, mask_path, path, dates):
        #dates = _AbstractInterpolator.interpolated_dates(self.timeline.study_dates)
        '''Render all volumes using supplied brain mask'''
        Parallel(n_jobs=multiprocessing.cpu_count())(
            delayed(Renderer._render_volume)(date, volume, path, overwrite=True)
            for date, volume in self.interpolator.interpolated_data_from_dates(files, mask_path, dates))

    def render_new(self, files, mask_path, path, dates):
        '''Render new volumes using supplied brain mask'''
        Parallel(n_jobs=multiprocessing.cpu_count())(
            delayed(Renderer._render_volume)(date, volume, path, overwrite=False)
            for date, volume in self.interpolator.interpolated_data_from_dates(files, mask_path, dates))

class VisTarsierRenderer(Renderer):
    def render_all(self, files, mask_path, path, dates):
        #dates = _AbstractInterpolator.interpolated_dates(self.timeline.study_dates)
        '''Render all volumes using supplied brain mask'''
        Parallel(n_jobs=multiprocessing.cpu_count())(
            delayed(Renderer._render_volume)(date, volume, path, overwrite=True)
            for date, volume in self.interpolator.interpolated_data_from_dates(files, mask_path, dates))

    def render_new(self, files, mask_path, path, dates):
        '''Render new volumes using supplied brain mask'''
        Parallel(n_jobs=multiprocessing.cpu_count())(
            delayed(Renderer._render_volume)(date, volume, path, overwrite=False)
            for date, volume in self.interpolator.interpolated_data_from_dates(files, mask_path, dates))
