import os
import subprocess
ANTSPATH = ''
if 'ANTSPATH' not in os.environ:
    print('***************** OH NO! ********************')
    print("Can't find ANTSPATH environment value. D:")
    print("pattools.ants is just a thin wrapper around a couple of ANTS tools that you need to install yourself.")
    print("We'll assume it's here somewhere (and in your PATH), but if I don't work try the docker image.")
    print('***************** /OH NO! *******************')
else:
    ANTSPATH = os.environ['ANTSPATH']

def n4_bias_correct(input, output):
    '''Calls N4BiasFieldCorrection'''
    p = subprocess.Popen([os.path.join(ANTSPATH,'N4BiasFieldCorrection'), '-i', input, '-o', output])
    return p

def affine_registration(floating, fixed, output):
    '''Calls antsRegistration'''
    p = subprocess.Popen([
        os.path.join(ANTSPATH,'antsRegistration'),
        '--dimensionality','3', # Run ANTS on 3 dimensional image
        '--float', '1',
        '--interpolation', 'Linear',
        '--use-histogram-matching', '0',
        '--initial-moving-transform', f'[{fixed},{floating},1]',
        '--transform', 'Affine[0.1]',
        '--metric', f'MI[{fixed},{floating},1,32,Regular,0.25]', # Use mutal information (we're not normalizing intensity)
        '--convergence', '[1000x500x250x100,1e-6,10]',
        '--shrink-factors', '8x4x2x1',
        '--smoothing-sigmas', '3x2x1x0vox',
        '--output', f'[{output}_,{output}]'
    ])
    return p

def syn_registration(floating, fixed, output, brain_mask=None):
    '''Calls antsRegistration'''
    p = subprocess.Popen([
        os.path.join(ANTSPATH,'antsRegistration'),
        '--dimensionality','3', # Run ANTS on 3 dimensional image
        '--float', '1',
        '--interpolation', 'Linear',
        '--use-histogram-matching', '0',
        '--initial-moving-transform', f'[{fixed},{floating},1]',
        '--transform', 'Affine[0.1]',
        '--metric', f'MI[{fixed},{floating},1,32,Regular,0.25]', # Use mutal information (we're not normalizing intensity)
        '--convergence', '[1000x500x250x100,1e-6,10]',
        '--shrink-factors', '8x4x2x1',
        '--smoothing-sigmas', '3x2x1x0vox',
        '--transform', 'SyN[0.1,3,0]',
        '--metric', f'CC[{fixed},{floating},1,4]',
        '--convergence', '[100x70x50x20,1e-6,10]',
        '--shrink-factors', '8x4x2x1',
        '--smoothing-sigmas', '3x2x1x0vox',
        '--output', f'[{output}_,{output}]'
    ])
    return p

def apply_transform(floating, fixed, affine, output, warp=None, invert_affine=False):
    '''Calls antsApplyTransforms'''
    # Basic command
    cmd = [
        os.path.join(ANTSPATH,'antsApplyTransforms'),
        '-d', '3',
        '--float', '1',
        '-i', floating,
        '-r', fixed,
        '-n', 'Linear']
    # Invert affine for inverse transforms
    if invert_affine:
        cmd.append('-t')
        cmd.append(f'[{affine},1]')
    else:
        cmd.append('-t')
        cmd.append(affine)
    # Apply warp if included
    if warp != None:
        cmd.append('-t')
        cmd.append(warp)
    # Output
    cmd.append('-o')
    cmd.append(output)

    p = subprocess.Popen(cmd)
    return p
