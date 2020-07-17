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

    @staticmethod
    def _date_from_path(path):
        datestring = os.path.basename(os.path.dirname(path))
        return date(int(datestring[0:4]), int(datestring[4:6]), int(datestring[6:]))

    @staticmethod
    def interpolated_dates(dates, delta_days):
        ''' Returns a set of interpolated dates from a given list of dates and delta in days.'''
        result_dates = []
        for date1, date2 in zip(dates[:-1], dates[1:]):
            window = date2 - date1
            steps = int(window.days / delta_days)
            for i in range(0, steps):
                result_dates.append(date1+timedelta(days=int(i*delta_days)))
        # Add the last date
        result_dates.append(dates[-1])
        return result_dates


    def data_for_date(self, date, study_dates, study_paths, mask_data):
        ''' This method will return the interpolated data for a given date.
            We are assuming that the study_dates and study_paths are in the same order, which is sorted by date '''
        raise Exception("This is the base interpolator class. Use an implementation")
        

    def interpolated_data_from_dates(self, image_paths, mask_path, dates):
        study_dates = [_AbstractInterpolator._date_from_path(p) for p in image_paths]
        mask_data = None
        if mask_path != None and os.path.exists(mask_path):
            mask_data = nib.load(mask_path).get_fdata()
        else:
            print('mask path ' + str(mask_path) + ' does not exist, using no mask')
            mask_data = 1

        for date in dates:
            d, d2 = self.data_for_date(date, study_dates, image_paths, mask_data)
            yield d, d2

    def interpolated_data_from_delta(self, image_paths, mask_path, delta_days):
        ''' Returns a list of numpy volumes interpolated based on the delta days. All real scans are included.'''
        all_dates = _AbstractInterpolator.interpolated_dates([_AbstractInterpolator._date_from_path(p) for p in image_paths], delta_days)
        return self.interpolated_data_from_dates(image_paths, mask_path, all_dates)



class LinearInterpolator(_AbstractInterpolator):
    '''Interpolates data linearly'''
    def __init__(self):
        super().__init__()

    def interpolate(self, data1, data2, ratio):
        return data1 * (1-ratio) + data2 * ratio
    
    def data_for_date(self, date, study_dates, study_paths, mask_data):
        if date in study_dates:
            data = nib.load(study_paths[study_dates.index(date)]).get_fdata() * mask_data
            return (date, data)
        else:
            # Get the closest two study dates (before and after)
            np_study_dates = np.unique(np.asarray(study_dates))
            np_study_dates = np.argsort(np.abs(np_study_dates - date))
            idx_before = None
            idx_after = None
            for idx in np_study_dates:
                if idx_before == None and study_dates[idx] < date:
                    idx_before = idx
                elif idx_after == None and study_dates[idx] > date:
                    idx_after = idx
            
            # If the date doesn't fall between two dates it's going to be meaningless
            if (idx_before == None or idx_after == None):
                # So we'll just return a zero matrix based on the first entry
                return (date, nib.load(study_paths[0]).get_fdata() * 0)

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
                ratio = delta / after_delta
            # yield the interpolated result
            return (date, self.interpolate(before_data, after_data, ratio))



class NearestNeighbourInterpolator(LinearInterpolator):
    '''Returns the nearest real scan'''
    def __init__(self):
        super().__init__()

    def interpolate(self, data1, data2, ratio):
        if ratio >= 0.5: return data2
        return data1

class NullInterpolator(LinearInterpolator):
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

    def render(self, timeline, path, overwrite=True):
        '''Write images to path given based on a timeline. Files will be interpolated and rendered to <path>/<filter>/<cor|sag|ax>/<date>/'''
        print('timeline.path', timeline.path)
        print('timeline.whitematter_mask', timeline.whitematter_mask)
        mask_path = None
        whitematter_mask_path = None
        if timeline.path != None and timeline.brain_mask != None:
             mask_path = os.path.join(timeline.path, timeline.brain_mask)
        if timeline.whitematter_mask != None:
            whitematter_mask_path = os.path.join(timeline.path, timeline.whitematter_mask)

        for filter in timeline.filters:
            files = Renderer._get_files(timeline, filter)
            self.render_all(
                files,
                mask_path, 
                os.path.join(path, filter.name), 
                timeline.study_dates(), 
                whitematter_mask_path=whitematter_mask_path,
                overwrite=overwrite)

    def render_new_studies(self, timeline, path):
        '''Write images to path given based on a timeline. Files will be interpolated and rendered to <path>/<filter>/<cor|sag|ax>/<date>/'''
        self.render(timeline, path, overwrite=False)

    @staticmethod
    def write_images(data, folder, slice_type, min_val, max_val):
        if not os.path.exists(folder):
            os.makedirs(folder, mode=0o777)
        #data_cp = np.copy(data)
        data_cp = np.flip(data, 0)
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
        #np.clip(output, min, max)
        imageio.imwrite(location, output)

    @staticmethod
    def _render_volume(date, volume, path, overwrite=True, whitematter_mask=1):
        '''Render every slice in a volume along 3 axis.'''
        # Set the min to be 0
        print('np.amin(volume)', np.amin(volume))
        volume = volume - np.amin(volume)
        # Make sure the curve falls between 0 and 400
        max_val = np.amax(volume)
        print('max_val', max_val)
        if max_val > 255:
            volume = volume / max_val * 255
        # Make the whitematter median value 127
        #if type(whitematter_mask) != type(int): #i.e. check that it's an ndarray but too lazy to look up syntax at 7:20pm
        #    wm_volume = volume * whitematter_mask
        #    wm_median = np.mean(wm_volume)
        #    print('wm_median', wm_median)
        #    wm_delta = min(255, max_val) / 15 - wm_median # This is the delta between the desired value (half of the bitspace) and the current value.
        #    print('wm_delta', wm_delta)
        #    volume = volume + wm_delta
        # Set the min/max to a valid range
        #np.clip(volume, 0, 255, volume)
        volume = volume.astype(np.uint8)

        # Output paths
        sag_path = os.path.join(path, date.strftime('%Y%m%d'), 'sag')
        cor_path = os.path.join(path, date.strftime('%Y%m%d'), 'cor')
        ax_path = os.path.join(path, date.strftime('%Y%m%d'), 'ax')
        # Write images if the folder doesn't exist or overwrite is true
        if overwrite or os.path.exists(sag_path) == False:
            Renderer.write_images(volume, sag_path, 'sag', 0, 255)
        if overwrite or os.path.exists(cor_path) == False:
            Renderer.write_images(volume, cor_path, 'cor', 0, 255)
        if overwrite or os.path.exists(ax_path) == False:
            Renderer.write_images(volume, ax_path, 'ax', 0, 255)

    def render_all(self, files, mask_path, path, dates, whitematter_mask_path=None, overwrite=True):
        # Load the whitematter mask to use (or use 1, i.e. no mask if none is found)
        whitematter_mask = 1
        print('??whitematter_mask_path', whitematter_mask_path)
        if whitematter_mask_path !=  None:
            whitematter_mask = nib.load(whitematter_mask_path).get_fdata()

        '''Render all volumes using supplied brain mask'''
        Parallel(n_jobs=multiprocessing.cpu_count())(
            delayed(Renderer._render_volume)(date, volume, path, overwrite=overwrite, whitematter_mask=whitematter_mask)
            for date, volume in self.interpolator.interpolated_data_from_dates(files, mask_path, dates))

    def render_new(self, files, mask_path, path, dates,  whitematter_mask_path=None):
        self.render_all(files, mask_path, path, dates, whitematter_mask_path=whitematter_mask_path, overwrite=False)

#class VisTarsierRenderer(Renderer):
#    def render_all(self, files, mask_path, path, dates, whitematter_mask_path=None, overwrite=True):
#        #dates = _AbstractInterpolator.interpolated_dates(self.timeline.study_dates)
#        '''Render all volumes using supplied brain mask'''
#        Parallel(n_jobs=multiprocessing.cpu_count())(
#            delayed(Renderer._render_volume)(date, volume, path, overwrite=True)
#            for date, volume in self.interpolator.interpolated_data_from_dates(files, mask_path, dates))
