import os
from enum import Enum
import json
import nibabel as nib

SUBJECT_PREFIX = 'sub-'
SESSION_PREFIX = 'ses-'
BIDS_VERSION = 'v1.6.0'

class ModalityType(Enum):
    ANATOMICAL = 'anat'
    FUNCTIONAL = 'fmri'
    FMAP = 'fmap'
    MEG = 'meg'
    IEEG = 'ieeg'
    DWI = 'dwi'
    PET = 'pet'

    def values():
        return [k.value for k in ModalityType]


class SourceDatasetDescription:
    URL: str = None
    DOI: str = None
    Version: str = None

    def from_dict(dct: dict):
        if dct is None:
            return None
        obj = SourceDatasetDescription()
        obj.URL = dct.get('URL', None)
        obj.DOI = dct.get('DOI', None)
        obj.Version = dct.get('Version', None)

        return obj

    def to_dict(self):
        return {
            'URL': self.URL,
            'DOI': self.DOI,
            'Version': self.Version
        }

class DatasetDescription:
    path: str
    parent = None
    #BIDS defined fields
    Name: str = None#REQUIRED
    BIDSVersion: str = None#REQUIRED
    HEDVersion: str = None#RECOMMENDED
    DatasetType: str = None#RECOMMENDED [raw|derivative]
    License: str = None#RECOMMENDED
    Authors: [str] = []#OPTIONAL
    Acknowledgements: str = None#OPTIONAL
    HowToAcknowledge: str = None#OPTIONAL
    Funding: [str] = []#OPTIONAL
    EthicsApprovals: [str] = [] #OPTIONAL
    ReferencesAndLinks: [str] = [] #OPTIONAL
    DatasetDOI: str = None #OPTIONAL
    #BIDS defined fields for derived types
    GeneratedBy = []
    SourceDatasets = []

    def __init__(self, name):
        self.BIDSVersion = BIDS_VERSION
        self.Name = name
    
    def load(path):
        with open(path, 'r') as f:
            dd = DatasetDescription.from_dict(json.loads(f.read()))
            dd.path = path
            return dd

    def save(self):
        # Technically this an empty name is a name...
        if self.Name == None:
            self.Name = ''
        if self.BIDSVersion == None:
            self.BIDSVersion = BIDS_VERSION

        with open(self.path, 'w') as f:
            f.write(json.dumps(self.to_dict()))

    def from_dict(dct):
        obj = DatasetDescription(dct.get('Name', None))        
        obj.BIDSVersion = dct.get('BIDSVersion', None)
        obj.HEDVersion = dct.get('HEDVersion', None)
        obj.DatasetType = dct.get('DatasetType', None)
        obj.License = dct.get('License', None)
        obj.Authors = dct.get('Authors', None)
        obj.Acknowledgements = dct.get('Acknowledgements', None)
        obj.HowToAcknowledge = dct.get('HowToAcknowledge', None)
        obj.EthicsApprovals = dct.get('EthicsApprovals', None)
        obj.ReferencesAndLinks = dct.get('ReferencesAndLinks', None)
        obj.DatasetDOI = dct.get('DatasetDOI', None)
        if 'GeneratedBy' in dct:
            obj.GeneratedBy = []
            for d in dct['GeneratedBy']:
                gb = GeneratedByDescription.from_dict(d)
                gb.parent = obj
                obj.GeneratedBy.append(gb)
        if 'SourceDatasets' in dct:
            obj.SourceDatasets = []
            for d in dct['SourceDatasets']:
                sd = SourceDatasetDescription.from_dict(d)
                sd.parent = obj
                obj.SourceDatasets.append(sd)
        return obj

    def to_dict(self):
        print(self.HEDVersion)
        return {
            'Name': self.Name,
            'BIDSVersion': self.BIDSVersion,
            'HEDVersion': self.HEDVersion,
            'DatasetType': self.DatasetType,
            'License': self.License,
            'Authors': json.dumps(self.Authors),
            'Acknowledgements': self.Acknowledgements,
            'HowToAcknowledge': self.HowToAcknowledge,
            'Funding': json.dumps(self.Funding),
            'EthicsApprovals': json.dumps(self.EthicsApprovals),
            'ReferencesAndLinks': json.dumps(self.ReferencesAndLinks),
            'DatasetDOI': self.DatasetDOI,
            'GeneratedBy': [o.to_dict() for o in self.GeneratedBy],
            'SourceDatasets': [o.to_dict() for o in self.SourceDatasets]
        }

class Dataset:
    parent = None
    path: str
    description: DatasetDescription
    
    def __init__(self, path):
        self.path = path
        try:
            self.description = DatasetDescription.load(os.path.join(path, 'dataset_description.json'))
        except:
            self.description = DatasetDescription(os.path.split(path)[1])
            self.description.path = os.path.join(path, 'dataset_description.json')
            self.description.save()
    
    def subjects(self):
        root, folders, _ = next(os.walk(self.path))
        # Subject folders are "sub-<label>".
        return [Subject(os.path.join(root, f), self) for f in folders if f.startswith(SUBJECT_PREFIX)]

class ContainerDescription:
    parent = None
    #BIDS defined fields
    Type: str #e.g. docker
    Tag: str #e.g. plwp/pattools
    URI: str 

    def from_dict(dct: dict):
        if dct is None:
            return None
        obj = ContainerDescription()
        obj.Type = dct.get('Type', None)
        obj.Tag = dct.get('Tag', None)
        obj.URI = dct.get('URI', None)
        return obj
    
    def to_dict(self):
        return {
            'Type': self.Type,
            'Tag': self.Tag,
            'URI': self.URI
        }     

class GeneratedByDescription:
    parent: DatasetDescription
    #BIDS defined fields
    Name: str = None #REQUIRED
    Version: str = None #RECOMMENDED
    Description: str = None #OPTIONAL
    CodeURL: str = None #OPTIONAL
    Container: ContainerDescription = None #OPTIONAL

    def from_dict(dct: dict):
        if dct is None:
            return None
        obj = GeneratedByDescription()
        obj.Name = dct.get('Name', None)
        obj.Version = dct.get('Version', None)
        obj.Description = dct.get('Description', None)
        obj.CodeURL = dct.get('CodeURL', None)
        if 'Container' in dct:
            obj.Container = ContainerDescription.from_dict(dct['Container'])
            if obj.Container != None:
                obj.Container.parent = obj
        return obj

    def to_dict(self):
        return {
            'Name': self.Name,
            'Version': self.Version,
            'Description': self.Description,
            'CodeURL': self.CodeURL,
            'Container': None if self.Container == None else self.Container.to_dict()
        }
   

class Subject:
    '''Represents a single subject within the dataset'''
    path: str = None
    label: str = None
    parent: Dataset = None

    def __init__(self, path:str, parent:Dataset=None):
        self.path = path
        self.parent = parent
        directory = os.path.split(path)[1]
        if not directory.startswith(SUBJECT_PREFIX):
            raise ValueError(f'{path} not a validly named BIDS subject folder')
        self.label = directory[4:]

    def sessions(self):
        '''Return all the sessions for this patient. If there are no session folders 
        the patient folder will be treated as a single session'''
        _, folders, _ = next(os.walk(self.path))
        session_folders = [f for f in folders if f.startswith(SESSION_PREFIX)]
        if len(session_folders) > 0:
            return [Session(os.path.join(self.path, s), self) for s in session_folders]
        else:
            # If there are no session subfolders then we can assume that the subject folder represents a single session.
            return [Session(self.path, self)]


class Session:
    '''Represents a single session folder'''
    path: str = None
    label: str = None
    parent: Subject = None

    def __init__(self, path:str, parent:Subject=None):
        self.path = path
        self.parent = parent
        directory = os.path.split(path)[1]
        if not directory.startswith(SESSION_PREFIX) and not directory.startswith(SUBJECT_PREFIX):
            raise ValueError(f'{path} not a validly named BIDS session or subject folder')
        if directory.startswith(SESSION_PREFIX):
            self.label = directory[4:]
        else:
            self.label = None # Empty label may be useful for parsing filenames
    
    def modalities(self):
        '''Return all the modalities for this session'''
        _, folders, _ = next(os.walk(self.path))
        input = [(os.path.join(self.path, f), self) for f in folders if f in ModalityType.values()]
        return [Modality(os.path.join(self.path, f), self) for f in folders if f in ModalityType.values()]

    def get_modality(self, modality_type: ModalityType):
        '''Return the modality based on type. May return None.'''
        modality_path = os.path.join(self.path, modality_type.value)
        if os.path.exists(modality_path):
            return Modality(modality_path, self)
        # TODO: There may be a better error here.
        return None

class ScanFile:
    path: str = None
    parent = None
    metadata_path: str = None

    def __init__(self, path:str, parent=None):
        self.path = path
        self.parent = parent
        if path.endswith('.nii.gz'):
            core = path[:-7]
        else:
            core = os.path.splitext(path)[0]
        self.metadata_path = core + '.json'

    def read_metadata(self):
        if os.path.exists(self.metadata_path):
            with open(self.metadata_path, 'r') as f:
                return json.load(f)

    def write_metadata(self, dct:dict):
        with open(self.metadata_path, 'w') as f:
            json.dump(dct, f)
        
    def read_image(self):
        return nib.load(self.path)

    def write_image(self, img):
        nib.save(img, self.path)

class Modality:
    path: str = None
    parent: Session = None
    modality_type: ModalityType = None

    def __init__(self, path:str, parent:Session=None):
        self.path = path
        self.parent = parent
        modality_type = ModalityType(os.path.split(self.path)[1])

    def scans(self):
        _, _, files = next(os.walk(self.path))
        return [ScanFile(os.path.join(self.path, f)) for f in files if f.endswith('.nii.gz') or f.endswith('.nii')]




