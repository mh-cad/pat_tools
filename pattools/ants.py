import os
ANTSPATH = ''
if 'ANTSPATH' not in os.environ:
    print('***************** OH NO! ********************')
    print("Can't find ANTSPATH environment value. D:")
    print("pattools.ants is just a thin wrapper around a couple of ANTS tools that you need to install yourself.")
    print("We'll assume it's here somewhere (and in your PATH), but if I don't work try the docker image.")
    print('***************** /OH NO! *******************')
else:
    ANTSPATH = os.environ['ANTSPATH']
