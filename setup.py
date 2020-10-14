import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="pattools",
    version="0.0.2",
    author="plwp",
    author_email="pat@plwp.net",
    description="Toolkit for neuro-imaging data manipulation and automation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/plwp/pat_tools",
    packages=['pattools'],
    install_requires=[
          'pydicom',
          'pynetdicom',
          'dicom2nifti',
          'requests',
          'nibabel',
          'clint',
          'imageio',
          'joblib'
      ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
