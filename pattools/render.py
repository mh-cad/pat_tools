from scipy.ndimage import affine_transform
from scipy import ndimage
import numpy as np
import imageio
#import math
#from PIL import Image, ImageEnhance

# The identity matrix for reference.
I = [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]

# Where X is defining the Left/Right axis
def rotation_X(A):
    theta = np.arccos(A[1][1])
    return np.degrees(theta)

# Where Y is defining the Posterior/Anterior (bum to nose) axis
def rotation_Y(A):
    theta = np.arccos(A[0][0])
    return np.degrees(theta)

# Where Z is defining the Inferior/Superior (feet to head) axis
def rotation_Z(A):
    theta = np.arccos(A[0][0])
    return np.degrees(theta)

def align(data, affine):
    # Align Y axis by rotating around the X axis
    data = ndimage.rotate(data, -rotation_X(affine), axes=(1,2),reshape=True)
    # Align X axis by rotating around the Y axis
    data = ndimage.rotate(data, -rotation_Y(affine), axes=(0,2),reshape=True)

    # Reflect the X alignment in the affine
    affine[0][1] = affine[0][2] = 0
    affine[1][0] = affine[2][0] = 0
    # Reflect the Y alignment in the affine
    affine[1][2] = affine[2][1] = 0

    return data, affine

def write_images(data, folder, slice_type, min_val, max_val):
    count = 0
    if slice_type == 'sag':
        count = data.shape[0]
        for i in range(data.shape[0]):
            write_image(data[i,:,:], os.path.join(folder, f'{i}.png'), min_val, max_val)

    elif slice_type == 'cor':
        count = data.shape[1]
        for j in range(data.shape[1]):
            write_image(data[:,j,:], os.path.join(folder, f'{j}.png'), min_val, max_val)

    elif slice_type == 'ax':
        count = data.shape[2]
        for k in range(data.shape[2]):
            write_image(data[:,:,k], os.path.join(folder, f'{k}.png'), min_val, max_val)

    return count

def write_image(slice, location, min, max):
    # This is a bit of a hack to make sure the range is normal
    slice[0,0] = max
    slice[0,1] = min
    output = np.flip(slice.T).copy()
    np.clip(output, min, max)
    imageio.imwrite(location, output)

def dodgyraytrace(line):
    # Get the surface value (anything above a 10)
    idx = np.argmax(line > 80)
    # Penetration depth
    #for i in range(0,3):
    #    idx+=1
    #    idx += np.argmax(line[idx:] > 80)

    val = line[idx]
    # Add some really snazzy lighting effects!
    if idx < 1: return 0
    val = (val / (math.log(idx)+1))
    if val < 0: return 0
    return val
