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
    p = subprocess.Popen([os.path.join(ANTSPATH,'N4BiasFieldCorrection'), '-i', input, '-o', output])
    return p

def affine_registration(floating, fixed, output):
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

def apply_linear_transform(floating, fixed, matrix, output):
    p = subprocess.Popen([
        os.path.join(ANTSPATH,'antsApplyTransforms'),
        '-d', '3',
        '--float', '1',
        '-i', floating,
        '-r', fixed,
        '-o', output,
        '-n', 'Linear',
        '-t', matrix])

    return p
