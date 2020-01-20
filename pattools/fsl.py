import os
FSLPATH = ''
if 'FSLPATH' not in os.environ:
    print('***************** OH NO! ********************')
    print("Can't find FSLPATH environment value. D:")
    print("pattools.fsl is just a thin wrapper around a couple of FSL tools that you need to install yourself.")
    print("We'll assume it's here somewhere (and in your PATH), but if I don't work try the docker image.")
    print('***************** /OH NO! *******************')
else:
    FSLPATH = os.environ['FSLPATH']
