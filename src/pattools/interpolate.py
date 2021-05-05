import os
from datetime import date, timedelta, datetime
import nibabel as nib
import numpy as np
import imageio
from PIL import Image
from joblib import Parallel, delayed
import multiprocessing
from pattools.image import m_mode

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
        '''This method will interpolate data for every data given in dates.
        The method will assume the image paths are formatted within a Timeline (so the date
        for each image can be extracted via the _AbstractInterpolator._date_from_path method.'''
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
        ''' This method will interpolate data based on images with a time between returned images
        that is delta_days days'''
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

    def render(self, timeline, path, overwrite=True, output_type='gray16'):
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
                overwrite=overwrite,
                output_type=output_type)

    def render_new_studies(self, timeline, path, output_type='gray16'):
        '''Write images to path given based on a timeline. Files will be interpolated and rendered to <path>/<filter>/<cor|sag|ax>/<date>/'''
        self.render(timeline, path, overwrite=False, output_type=output_type)

    @staticmethod
    def write_images(data, folder, slice_type, min_val, max_val, image_mode='gray16-rgbstuffed'):
        print('writing images...')
        if not os.path.exists(folder):
            os.makedirs(folder, mode=0o777)

        data_cp = np.flip(data, 0)
        count = 0
        if slice_type == 'sag':
            count = data.shape[0]
            for i in range(data.shape[0]):
                Renderer.write_image(data_cp[i,:,:], os.path.join(folder, f'{i}.png'), min_val, max_val, image_mode)

        elif slice_type == 'cor':
            count = data.shape[1]
            for j in range(data.shape[1]):
                Renderer.write_image(data_cp[:,j,:], os.path.join(folder, f'{j}.png'), min_val, max_val, image_mode)

        elif slice_type == 'ax':
            count = data.shape[2]
            for k in range(data.shape[2]):
                Renderer.write_image(data_cp[:,:,k], os.path.join(folder, f'{k}.png'), min_val, max_val, image_mode)

        return count

    @staticmethod
    def write_image(slice, location, min, max, mode='gray8'):
        '''Write a single slice to an image at the given location'''
        if mode == 'gray16':
            Renderer.write_grayscale16_image(slice, location, min, max)
        elif mode == 'gray8':
            Renderer.write_grayscale8_image(slice, location, min, max)
        elif mode == 'gray16-rgbstuffed':
            Renderer.write_grayscale16_rgbstuffed_image(slice, location, min, max)

    @staticmethod
    def write_grayscale16_image(slice, location, min, max):
        '''Write a single slice to an image at the given location as 16bit greyscale'''
        # For non background slices we want to normalize the range to fit in uint16
        output = np.flip(slice.T).copy()
        output = output.astype(np.uint16)
        Image.fromarray(output).save(location)

    @staticmethod
    def write_grayscale16_rgbstuffed_image(slice, location, min, max):
        '''This will preserve the 16 bits of data to allow for windowing, but will output a standard RGB image.
        This method allows using standard libraries that expect RGB without losing data, but will need to be
        processed and windowed before output'''
        # For non background slices we want to normalize the range to fit in uint16
        output = np.flip(slice.T).copy()
        output = output.astype(np.uint16)

        rgbout = np.zeros((output.shape[0], output.shape[1], 3), np.uint8)
        rgbout[:,:,1] = (output >> 8).astype(np.uint8)
        
        rgbout[:,:,2] = (output - (rgbout[:,:,1] << 8)).astype(np.uint8)
        Image.fromarray(rgbout).save(location)

    @staticmethod
    def write_grayscale8_image(slice, location, min, max):
        '''Write a single slice to an image at the given location as 16bit greyscale'''
        # For non background slices we want to normalize the range to fit in uint16
        output = np.flip(slice.T).copy()
        output = output.astype(np.uint8)
        Image.fromarray(output).save(location)

    @staticmethod
    def _render_volume(date, volume, path, overwrite=True, whitematter_mask=1, output_type='gray16'):
        '''Render every slice in a volume along 3 axis.'''
        # Set the min to be 0
        volume = volume - np.amin(volume)
        # Make sure the curve falls between 0 and 400
        max_val = np.amax(volume)
        #if max_val > 255:
        #    volume = volume / max_val * 255
        #volume = volume.astype(np.uint8)

        # Output paths
        sag_path = os.path.join(path, date.strftime('%Y%m%d'), 'sag')
        cor_path = os.path.join(path, date.strftime('%Y%m%d'), 'cor')
        ax_path = os.path.join(path, date.strftime('%Y%m%d'), 'ax')
        # Write images if the folder doesn't exist or overwrite is true
        if overwrite or os.path.exists(sag_path) == False:
            Renderer.write_images(volume, sag_path, 'sag', 0, max_val)
        if overwrite or os.path.exists(cor_path) == False:
            Renderer.write_images(volume, cor_path, 'cor', 0, max_val)
        if overwrite or os.path.exists(ax_path) == False:
            Renderer.write_images(volume, ax_path, 'ax', 0, max_val)

    def render_all(self, files, mask_path, path, dates, whitematter_mask_path=None, overwrite=True, output_type='gray16'):
        # Load the whitematter mask to use (or use 1, i.e. no mask if none is found)
        whitematter_mask = 1
        print('??whitematter_mask_path', whitematter_mask_path)
        if whitematter_mask_path !=  None:
            whitematter_mask = nib.load(whitematter_mask_path).get_fdata()

        '''Render all volumes using supplied brain mask'''
        Parallel(n_jobs=multiprocessing.cpu_count())(
            delayed(Renderer._render_volume)(date, volume, path, overwrite=overwrite, whitematter_mask=whitematter_mask, output_type=output_type)
            for date, volume in self.interpolator.interpolated_data_from_dates(files, mask_path, dates))

    def render_new(self, files, mask_path, path, dates,  whitematter_mask_path=None, output_type='gray16'):
        self.render_all(files, mask_path, path, dates, whitematter_mask_path=whitematter_mask_path, overwrite=False, output_type=output_type)


#class VisTarsierRenderer(Renderer):
#    def render_all(self, files, mask_path, path, dates, whitematter_mask_path=None, overwrite=True):
#        #dates = _AbstractInterpolator.interpolated_dates(self.timeline.study_dates)
#        '''Render all volumes using supplied brain mask'''
#        Parallel(n_jobs=multiprocessing.cpu_count())(
#            delayed(Renderer._render_volume)(date, volume, path, overwrite=True)
#            for date, volume in self.interpolator.interpolated_data_from_dates(files, mask_path, dates))

class MMode():
    ''' M Mode renders a 2D view of the 4D timeline data between two spatial points (i.e. a line through time)'''
    timeline = None
    _start = None
    _end = None
    _lines = None
    
    def __init__(self, timeline, start, end):
        ''' M Mode renderer requires a timeline, start point and end point'''
        self.timeline = timeline
        timeline._load_datamap()
        self._start = start
        self._end = end

    def _calculate_lines(self, height=100):
        ''' This will give us a bunch of 1D lines, one for each scan in the timeline. '''
        if self._start == None or self._end == None:
            raise ValueError('Start and end points cannot be None')
        
        self._lines = {}
        for studydate in self.timeline.datamap:
            for fm in self.timeline.datamap[studydate]:
                path = os.path.join(self.timeline.path, studydate, fm.processed_file)
                try:
                    img = nib.load(path).get_fdata()
           
                    #resolve negative indexing (a little wasteful to do it inside the loop but we have an image shape and it's cheap)
                    for i in range(3):
                        if self._start[i] < 0:
                            self._start[i] += img.shape[i]
                        if self._end[i] < 0:
                            self._end[i] += img.shape[i]
                    # Calculate line         
                    line = m_mode(img, self._start, self._end, out_size=height)
                    if fm.filter_name not in self._lines:
                        self._lines[fm.filter_name] = {}
                    self._lines[fm.filter_name][studydate] = line
                except Exception as e:
                    print(f'Something went wrong with {fm.filter_name}/{studydate}')
                    print(e)
        return self._lines

    def _parse_date(self, date_time_str):
        ''' Converts a timeline date into a timeline for norming. '''
        return datetime.strptime(date_time_str, '%Y%m%d').timestamp()

    # TODO: This function is trying to do two things (output the file and return the dict), they should be separated.
    def render(self, filter_name, outfile=None, line_width=5, width=400, height=100):
        ''' This function will render a 2D view of a given filter to the outfile.
            Background will be black in an image defined by height and width.
            Return value is a list of dictionaries with normed dates and line values. '''
        # Calculate the lines as needed
        if self._lines == None or len(self._lines.keys()) == 0 or self._lines[next(self._lines.keys())].shape[1] != height:
            self._calculate_lines(height)
                
        # Check input
        if self._lines == None or self._start == None or self._end == None:
            raise ValueError('Points have not been set correctly')
        if filter_name not in self._lines or len(self._lines[filter_name].keys()) == 0:
            raise ValueError("Filter doesn't seem to exist")
        
        lines = self._lines[filter_name]
        
        outdata = np.zeros((width, height))
        dates = []
        
        # Parse the date values
        max_date = float('-inf')
        min_date = float('inf')
        for date_string in lines.keys():
            d = self._parse_date(date_string)
            dates.append({'date_string':date_string, 'date':d, 'line':lines[date_string]})
            min_date = min(min_date, d)
            max_date = max(max_date, d)
        
        # Write the line to the output image
        max_date -= min_date
        for ddict in dates:
            normed_date = (ddict['date'] - min_date) / max_date
            index = int(normed_date * (width-line_width))
            for i in range(line_width):
                if index+i < outdata.shape[0] and index+1 >= 0:
                    outdata[index+i,:] = ddict['line']
        
        # Write out file
        if outfile != None:
            # Norm data to 16bit signed int
            outdata -= np.min(outdata)
            maxval = np.max(outdata)
            if maxval != 0:
                outdata /= maxval
                outdata *= 255
            
            outimg = Image.fromarray(outdata.T.astype(np.uint8), mode='L')#, mode='I;16')
            outimg.save(outfile)
            print(np.array(outimg))

        return dates
