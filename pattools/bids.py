import os
from enum import Enum

SUBJECT_PREFIX = 'sub-'
SESSION_PREFIX = 'ses-'

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

class Dataset:
    path: str
    
    def __init__(self, path):
        self.path = path
    
    def subjects(self):
        root, folders, _ = next(os.walk(self.path))
        # Subject folders are "sub-<label>".
        return [Subject(os.path.join(root, f), self) for f in folders if f.startswith(SUBJECT_PREFIX)]

class Subject:
    path: str
    label: str
    parent: Dataset

    def __init__(self, path:str, parent:Dataset=None):
        self.path = path
        self.parent = parent
        directory = os.path.split(path)[1]
        if not directory.startswith(SUBJECT_PREFIX):
            raise ValueError(f'{path} not a validly named BIDS subject folder')
        self.label = directory[4:]

    def sessions(self):
        _, folders, _ = next(os.walk(self.path))
        session_folders = [f for f in folders if f.startswith(SESSION_PREFIX)]
        if len(session_folders) > 0:
            return [Session(os.path.join(self.path, s), self) for s in session_folders]
        else:
            # If there are no session subfolders then we can assume that the subject folder represents a single session.
            return [Session(self.path, self)]


class Session:
    path: str
    label: str
    parent: Subject

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
        _, folders, _ = next(os.walk(self.path))
        [Modality(os.path.join(f, self.path), self) for f in folders if f in ModalityType.values()]

class Modality:
    path: str
    modality_type: ModalityType

    def __init__(self, path:str, parent:Session=None):
        self.path = path
        modality_type = ModalityType(path.split(self.path)[1])



