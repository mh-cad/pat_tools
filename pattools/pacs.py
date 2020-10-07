'''The PACs module contains classes that enable simple, patient-based access to PACS data.'''

import ast
import os
from dicom2nifti.convert_dicom import dicom_array_to_nifti
from socket import gethostname
from pydicom.dataset import Dataset
from pynetdicom import (
    AE, evt, build_role,
    PYNETDICOM_IMPLEMENTATION_UID,
    PYNETDICOM_IMPLEMENTATION_VERSION)
# pylint: disable=no-name-in-module
# (these are generated at runtime for this module)
from pynetdicom.sop_class import (
    PatientRootQueryRetrieveInformationModelFind,
    StudyRootQueryRetrieveInformationModelFind,
    StudyRootQueryRetrieveInformationModelGet,
    # Storage classes
    ComputedRadiographyImageStorage,
    DigitalXRayImagePresentationStorage,
    DigitalXRayImageProcessingStorage,
    DigitalMammographyXRayImagePresentationStorage,
    DigitalMammographyXRayImageProcessingStorage,
    CTImageStorage,
    EnhancedCTImageStorage,
    LegacyConvertedEnhancedCTImageStorage,
    UltrasoundMultiframeImageStorage,
    MRImageStorage,
    EnhancedMRImageStorage,
    MRSpectroscopyStorage,
    EnhancedMRColorImageStorage,
    LegacyConvertedEnhancedMRImageStorage,
    UltrasoundImageStorage,
    GrayscaleSoftcopyPresentationStateStorage,
    ColorSoftcopyPresentationStateStorage,
    PseudocolorSoftcopyPresentationStageStorage,
    BlendingSoftcopyPresentationStateStorage,
    XAXRFGrayscaleSoftcopyPresentationStateStorage,
    GrayscalePlanarMPRVolumetricPresentationStateStorage,
    CompositingPlanarMPRVolumetricPresentationStateStorage,
    VolumeRenderingVolumetricPresentationStateStorage,
    SegmentedVolumeRenderingVolumetricPresentationStateStorage,
    MultipleVolumeRenderingVolumetricPresentationStateStorage,
    XRayAngiographicImageStorage,
    EnhancedXAImageStorage,
    XRayRadiofluoroscopicImageStorage,
    EnhancedXRFImageStorage,
    XRay3DAngiographicImageStorage,
    XRay3DCraniofacialImageStorage,
    BreastTomosynthesisImageStorage,
    BreastProjectionXRayImagePresentationStorage,
    BreastProjectionXRayImageProcessingStorage,
    IntravascularOpticalCoherenceTomographyImagePresentationStorage,
    IntravascularOpticalCoherenceTomographyImageProcessingStorage,
    NuclearMedicineImageStorage,
    ParametricMapStorage,
    RawDataStorage,
    SpatialRegistrationStorage,
    SpatialFiducialsStorage,
    DeformableSpatialRegistrationStorage,
    SegmentationStorage,
    SurfaceSegmentationStorage,
    TractographyResultsStorage,
    RealWorldValueMappingStorage,
    SurfaceScanMeshStorage,
    SurfaceScanPointCloudStorage,
    VLEndoscopicImageStorage,
    VideoEndoscopicImageStorage,
    VLMicroscopicImageStorage,
    VideoMicroscopicImageStorage,
    VLSlideCoordinatesMicroscopicImageStorage,
    VLPhotographicImageStorage,
    VideoPhotographicImageStorage,
    OphthalmicPhotography8BitImageStorage,
    OphthalmicPhotography16BitImageStorage,
    StereometricRelationshipStorage,
    OphthalmicTomographyImageStorage,
    WideFieldOphthalmicPhotographyStereographicProjectionImageStorage,
    WideFieldOphthalmicPhotography3DCoordinatesImageStorage,
    OphthalmicOpticalCoherenceTomographyEnFaceImageStorage,
    OphthlamicOpticalCoherenceTomographyBScanVolumeAnalysisStorage,
    VLWholeSlideMicroscopyImageStorage,
    LensometryMeasurementsStorage,
    AutorefractionMeasurementsStorage,
    KeratometryMeasurementsStorage,
    SubjectiveRefractionMeasurementsStorage,
    VisualAcuityMeasurementsStorage,
    SpectaclePrescriptionReportStorage,
    OphthalmicAxialMeasurementsStorage,
    IntraocularLensCalculationsStorage,
    MacularGridThicknessAndVolumeReport,
    OphthalmicVisualFieldStaticPerimetryMeasurementsStorage,
    OphthalmicThicknessMapStorage,
    CornealTopographyMapStorage,
    BasicTextSRStorage,
    EnhancedSRStorage,
    ComprehensiveSRStorage,
    Comprehensive3DSRStorage,
    ExtensibleSRStorage,
    ProcedureSRStorage,
    MammographyCADSRStorage,
    KeyObjectSelectionStorage,
    ChestCADSRStorage,
    XRayRadiationDoseSRStorage,
    RadiopharmaceuticalRadiationDoseSRStorage,
    ColonCADSRStorage,
    AcquisitionContextSRStorage,
    SimplifiedAdultEchoSRStorage,
    PositronEmissionTomographyImageStorage,
    EnhancedPETImageStorage,
    LegacyConvertedEnhancedPETImageStorage,
    BasicStructuredDisplayStorage,
    RTImageStorage
    )


def cget_series(scp_settings, dataset):
    '''Runs a series level CGET request'''
    ae = AE(scp_settings.local_aet)
    # Add the requested presentation contexts (QR SCU)
    ae.add_requested_context(StudyRootQueryRetrieveInformationModelGet)
    # Add the requested presentation context (Storage SCP)
    ae.add_requested_context(CTImageStorage)
    ae.add_requested_context(EnhancedCTImageStorage)
    ae.add_requested_context(MRImageStorage)
    ae.add_requested_context(MRSpectroscopyStorage)
    ae.add_requested_context(EnhancedMRColorImageStorage)
    # Note, we can add to these if we want other types of image (see massive import statement)

    # Create an SCP/SCU Role Selection Negotiation item for Image Storage
    roles = []
    roles.append(build_role(CTImageStorage, scp_role=True))
    roles.append(build_role(EnhancedCTImageStorage, scp_role=True))
    roles.append(build_role(MRImageStorage, scp_role=True))
    roles.append(build_role(MRSpectroscopyStorage, scp_role=True))
    roles.append(build_role(EnhancedMRColorImageStorage, scp_role=True))

    dicoms = []
    def handle_store_series(e):
        nonlocal dicoms

        ds = e.dataset
        context = e.context
        meta = Dataset()
        # Add the DICOM File Meta Information
        meta = Dataset()
        meta.MediaStorageSOPClassUID = ds.SOPClassUID
        meta.MediaStorageSOPInstanceUID = ds.SOPInstanceUID
        meta.ImplementationClassUID = PYNETDICOM_IMPLEMENTATION_UID
        meta.ImplementationVersionName = PYNETDICOM_IMPLEMENTATION_VERSION
        meta.TransferSyntaxUID = context.transfer_syntax

        # Add the file meta to the dataset
        ds.file_meta = meta

        # Set the transfer syntax attributes of the dataset
        ds.is_little_endian = context.transfer_syntax.is_little_endian
        ds.is_implicit_VR = context.transfer_syntax.is_implicit_VR

        dicoms.append(ds)

        return 0x0000

    handlers = [(evt.EVT_C_STORE, handle_store_series)]

    # Associate with peer AE
    assoc = ae.associate(
        scp_settings.host, scp_settings.port,
        ae_title=scp_settings.ae_title, ext_neg=roles, evt_handlers=handlers)

    # We're (apparently) using the handle_store_series 
    if assoc.is_established:
        responses = assoc.send_c_get(dataset, StudyRootQueryRetrieveInformationModelGet)
        for (status, _) in responses:
            if status:
                pass
            else:
                raise TimeoutError('Connection timed out, was aborted or received invalid response')
        assoc.release()
        return dicoms
    else:
        raise ConnectionError('Association rejected, aborted or never connected')

def cget_report(scp_settings, dataset):
    '''Runs a CGET request for a report'''
    ae = AE(scp_settings.local_aet)
    # Add the requested presentation contexts (QR SCU)
    ae.add_requested_context(StudyRootQueryRetrieveInformationModelGet)
    # Add the requested presentation context (Storage SCP)
    ae.add_requested_context(BasicTextSRStorage)
    ae.add_requested_context(EnhancedSRStorage)

    # Create an SCP/SCU Role Selection Negotiation item for CT Image Storage
    roles = []
    roles.append(build_role(BasicTextSRStorage, scp_role=True))
    roles.append(build_role(EnhancedSRStorage, scp_role=True))

    reportds = None
    def handle_store_report(e):
        nonlocal reportds
        reportds = e.dataset
        return  0x0000

    handlers = [(evt.EVT_C_STORE, handle_store_report)]

    # Associate with peer AE
    assoc = ae.associate(
        scp_settings.host, scp_settings.port,
        ae_title=scp_settings.ae_title, ext_neg=roles, evt_handlers=handlers)

    if assoc.is_established:
        responses = assoc.send_c_get(dataset, StudyRootQueryRetrieveInformationModelGet)
        for (status, _) in responses:
            if status:
                pass
            else:
                raise TimeoutError('Connection timed out, was aborted or received invalid response')

        assoc.release()
        return reportds

    else:
        raise ConnectionError('Association rejected, aborted or never connected')

def cfind(scp_settings, dataset, query_model='P'):
    ''' Make a CFIND request '''
    ae = AE(scp_settings.local_aet)
    ae.add_requested_context(StudyRootQueryRetrieveInformationModelFind)
    ae.add_requested_context(PatientRootQueryRetrieveInformationModelFind)
    ae.add_requested_context('1.2.840.10008.5.1.4.1.1.88.11')
    assoc = ae.associate(
        scp_settings.host, scp_settings.port, ae_title=scp_settings.ae_title)
    # I'm sure send_c_find used to accept the single letter versions but it doesn't now.
    if query_model == 'S': query_model = StudyRootQueryRetrieveInformationModelFind
    elif query_model == 'P': query_model = PatientRootQueryRetrieveInformationModelFind

    #Check for successful association
    if assoc.is_established:
         # Use the C-FIND service to send the identifier
         responses = assoc.send_c_find(dataset, query_model=query_model)

         result = []
         for (status, identifier) in responses:
             if status:
                 # If the status is 'Pending' then identifier is the C-FIND response
                 if status.Status in (0xFF00, 0xFF01):
                     result.append(str(identifier))
             else:
                 assoc.release()
                 raise TimeoutError('Connection timed out, was aborted or received invalid response ' + scp_settings.ae_title)

         # Release the association
         assoc.release()
         return result
    else:
        raise ConnectionError('Association rejected, aborted or never connected ' + scp_settings.ae_title)

def find_studies_from_patient(patient):
    '''Runs a CFIND for all studies for a given patient'''
    ds = Dataset()
    ds.PatientID = patient.id
    ds.PatientName = patient.name
    ds.StudyInstanceUID = ''
    ds.AccessionNumber = ''
    ds.QueryRetrieveLevel = 'STUDY'
    ds.SeriesUID=''
    ds.PatientSize = ''
    ds.ModalitiesInStudy = ''
    ds.SOPClassesInStudy = ''
    ds.StudyDescription = '*'
    ds.FailedSOPSequence = ''
    ds.DerivationCodeSequence =''
    ds.SeriesInStudy=''
    ds.StudyDate=''
    ds.StudyTime=''
    results = cfind(patient.scp_settings, ds)
    studies = []
    for result in results:
        study = Study.parse_result(result)
        study.scp_settings = patient.scp_settings
        if study != None: studies.append(study)

    return studies

def find_series_from_study(study):
    '''CFINDS all series for a given study'''
    ds = Dataset()
    ds.StudyInstanceUID = study.study_uid
    ds.AccessionNumber = ''
    ds.QueryRetrieveLevel = 'SERIES'
    ds.SeriesInstanceUID=''
    ds.SeriesDescription = ''
    ds.SeriesNumber = ''
    ds.Modality =''
    ds.SeriesDate=''
    ds.StudyDate = ''
    ds.SeriesTime=''
    ds.StudyDescription = '*'

    results = cfind(study.scp_settings, ds, query_model='S')
    series = []
    for result in results:
        seri = Series.parse_result(result)
        seri.scp_settings = study.scp_settings
        if seri != None: series.append(seri)

    return series

#def find_studies(scp_settings, patient_name = '', patient_id = '' ,study_uid = '',
#    accession_number = '', modalities_in_study = '', study_date = '',
#    study_time = '', description = ''):
#    '''Run a study level CFIND requests using the supplied parameters'''
#    ds = Dataset()
#    ds.PatientID = patient_id
#    ds.PatientName = patient_name
#    ds.StudyInstanceUID = study_uid
#    ds.AccessionNumber = accession_number
#    ds.QueryRetrieveLevel = 'STUDY'
#    ds.ModalitiesInStudy = str(modalities_in_study)
#    ds.StudyDescription = description
#    ds.SeriesInStudy=''
#    ds.StudyDate=study_date
#    ds.StudyTime=study_time
#
#    results = cfind(scp_settings, ds)
#    studies = []
#    for result in results:
#        study = Study.parse_result(result)
#        study.scp_settings = patient.scp_settings
#        if study != None: studies.append(study)
#
#    return studies

### ScpSettings class
class ScpSettings:
  '''A class representing SCP settings (i.e. how to connect to PACS)'''
  ae_title = None
  host = None
  port = None
  local_aet = None

  def __init__(self, ae_title, host, port, local_aet=None):
      self.ae_title = ae_title
      self.host = host
      self.port = port
      if local_aet != None: self.local_aet = local_aet
      else: self.local_aet = ae_title
      # The above is a little sneaky because the remote PACS might not want to
      # talk to an unknown SCP. But it knows itself, so it will often be happy
      # to dump all its info to someone with the same name.

  def __str__(self):
      return (
        'ae_title[' + str(self.ae_title) +
        '], host[' + str(self.host) +
        '], port[' + str(self.port) + ']'
      )

### Patient class
class Patient:
    ''' Contains the details for a single patient '''
    id = None
    name = ''
    dob = ''
    patient_sex = ''
    scp_settings = None

    def __init__(self, id=None, scp_settings=None):
        ''' Requires an id string and the scp_settings of the remote PACs scp '''
        self.id = id
        self.scp_settings = scp_settings
        if id != None and scp_settings !=None:
            patient = Patient.from_id(id, scp_settings)
            if patient != None:
                self.name = patient.name
                self.dob = patient.dob
                self.patient_sex = patient.patient_sex
            else:
                print(f'Patient #{id} was not found in SCP')

    def __str__(self):
        return (
            'id          : ' + str(self.id) + '\n' +
            'name        : ' + str(self.name) + '\n' +
            'dob         : ' + str(self.dob) + '\n' +
            'patient_sex : ' + str(self.patient_sex) + '\n' +
            'scp_settings: ' + str(self.scp_settings))

    def find_studies(self):
        ''' Find all remote studies for the patient '''
        return find_studies_from_patient(self)

    @staticmethod
    def from_id(id, scp_settings):
        '''Returns a patient from a given patient id'''
        ds = Dataset()
        ds.PatientID = id
        ds.PatientName = ''
        ds.PatientBirthDate = ''
        ds.PatientSex = ''
        ds.QueryRetrieveLevel = 'PATIENT'
        results = cfind(scp_settings, ds)
        if len(results) > 0:
            return Patient.parse_result(results[0], scp_settings)
        else:
            return None

    @staticmethod
    def parse_result(result, scp_settings):
        ''' Parse a pynetdicom result and return a patient '''
        patient = Patient(scp_settings=scp_settings)
        for field in result.split('\n'):
            if field.startswith('(0010, 0010)'):
                patient.name = field.split()[-1].strip("'").strip('"')
            elif field.startswith('(0010, 0020)'):
                patient.id = field.split()[-1].strip("'").strip('"')
            elif field.startswith('(0010, 0040)'):
                patient.patient_sex = field.split()[-1].strip("'").strip('"')
            elif field.startswith('(0010, 0030)'):
                patient.dob = field.split()[-1].strip("'").strip('"')
        return patient
        
### Study class
class Study:
    ''' Contains the details for a single study'''
    study_uid = None
    patient_id = None
    accession_number = None
    modalities_in_study = None
    study_date = None
    study_time = None
    description = None
    scp_settings = None

    def __init__(self, scp_settings=None):
        self.scp_settings = scp_settings

    def __str__(self):
        return (
            'study_uid           : ' + str(self.study_uid) + '\n' +
            'accession_number    : ' + str(self.accession_number) + '\n' +
            'modalities_in_study : ' + str(self.modalities_in_study) + '\n' +
            'study_date          : ' + str(self.study_date) + '\n' +
            'study_time          : ' + str(self.study_time) + '\n' +
            'description         : ' + str(self.description)+ '\n' +
            'scp_settings        : ' + str(self.scp_settings))

    def find_series(self):
        '''Find all series for this study'''
        return find_series_from_study(self)

    def get_report(self):
        '''CGET the report for this study'''
        series = self.find_series()
        for seri in [s for s in series if s.modality == 'SR']:
            if seri.modality == 'SR':
                ds = Dataset()
                ds.SeriesInstanceUID = seri.series_uid
                ds.StudyInstanceUID = self.study_uid
                ds.QueryRetrieveLevel = 'SERIES'
                return Report(cget_report(self.scp_settings, ds))

                #for result in results:
                #    print("Result:::::::: ", result)
                #if (len(results) > 0):
                #    report = Report.parse_result(results[0])
                #    report.study = self
            print('could not find report')
            return None

    @staticmethod
    def parse_result(result):
        ''' Parse a pynetdicom result and return a study '''
        study = Study()
        for field in result.split('\n'):
            if field.startswith('(0008, 0020)'):
                study.study_date = field.split()[-1].strip("'").strip('"')
            elif field.startswith('(0008, 0030)'):
                study.study_time = field.split()[-1].strip("'").strip('"')
            elif field.startswith('(0008, 0050)'):
                study.accession_number = field.split(': ')[-1].strip("'").strip('"')
            elif field.startswith('(0008, 1030)'):
                study.description = field.split(': ')[-1].strip("'").strip('"')
            elif field.startswith('(0008, 0061)'):
                try:
                    study.modalities_in_study = ast.literal_eval(field.split(': ')[-1].strip("'").strip('"'))
                except:
                    study.modalities_in_study = [field.split(': ')[-1].strip("'").strip('"')]
            elif field.startswith('(0020, 000d)'):
                study.study_uid = field.split()[-1].strip("'").strip('"')
            elif field.startswith('(0010, 0020)'):
                study.patient_id = field.split()[-1].strip("'").strip('"')

        return study

class Series:
    '''Contains the details of a single series'''
    series_uid = None
    study_uid = None
    modality = None
    series_number = None
    description = None
    study_date = None
    scp_settings = None

    def __init__(self, scp_settings=None):
        self.scp_settings = scp_settings

    def __str__(self):
        return (
            'series_uid   : ' + str(self.series_uid) + '\n' +
            'study_uid     : ' + str(self.study_uid) + '\n' +
            'modality     : ' + str(self.modality) + '\n' +
            'series_number: ' + str(self.series_number) + '\n' +
            'description  : ' + str(self.description)+ '\n' +
            'study_date  : ' + str(self.study_date)+ '\n' +
            'scp_settings : ' + str(self.scp_settings))

    def save_dicom(self, dir_path):
        '''Save this series as a dicom (to folder)'''
        if not os.path.exists(dir_path): os.mkdir(dir_path)
        ds = Dataset()
        ds.SeriesInstanceUID = self.series_uid
        ds.StudyInstanceUID = self.study_uid
        ds.QueryRetrieveLevel = 'SERIES'
        dicoms = cget_series(self.scp_settings, ds)
        for dicom in dicoms:
             dicom.save_as(os.path.join(dir_path, dicom.SOPInstanceUID +'.dcm'), write_like_original=False)
        return True

    def save_nifti(self, file_path, max_attempts=5):
        '''Save this series as a nifti file'''
        print('saving file:', file_path)
        if not os.path.exists(os.path.dirname(file_path)) and os.path.dirname(file_path) != '':
            os.mkdir(os.path.dirname(file_path))
        ds = Dataset()
        ds.SeriesInstanceUID = self.series_uid
        ds.StudyInstanceUID = self.study_uid
        ds.QueryRetrieveLevel = 'SERIES'
        dicoms = None
        attempts = 0
        while dicoms == None and attempts < max_attempts:
            print('trying c_get')
            attempts += 1
            try:
                dicoms = cget_series(self.scp_settings, ds)
            except Exception as e:
                print(e)
                if attempts >= max_attempts:
                    raise
        try:
            dicom_array_to_nifti(dicoms, file_path, reorient_nifti=True)
            return True
        except:
            print('Failed to convert series to nifti, maybe try DICOM?')
            return False

    @staticmethod
    def parse_result(result):
        ''' Parse a pynetdicom result and return a series '''
        series = Series()
        for field in result.split('\n'):
            if field.startswith('(0020, 000e)'):
                series.series_uid = field.split()[-1].strip("'").strip('"')
            elif field.startswith('(0008, 0060)'):
                series.modality = field.split()[-1].strip("'").strip('"')
            elif field.startswith('(0020, 0011)'):
                series.series_number = field.split(': ')[-1].strip("'").strip('"')
            elif field.startswith('(0008, 103e)'):
                series.description = field.split(': ')[-1].strip("'").strip('"')
            elif field.startswith('(0020, 000d)'):
                series.study_uid = field.split()[-1].strip("'").strip('"')
            elif field.startswith('(0008, 0020)'):
                series.study_date = field.split(': ')[-1].strip("'").strip('"')
        return series

class Report(Series):
    '''This class represents a study report'''
    patient_name = None
    patient_id = None
    patient_dob = None
    patient_sex = None
    study_uid = None
    series_uid = None
    completion_flag = None
    verification_flag = None
    type = None
    text = None
    json_string = None

    def __init__(self, dataset=None, study=None):
            self.study = study
            if dataset != None:
                self.patient_name = dataset.PatientName
                self.patient_id = dataset.PatientID
                self.patient_dob = dataset.PatientBirthDate
                self.patient_sex = dataset.PatientSex
                self.study_uid = dataset.StudyInstanceUID
                self.series_uid = dataset.SeriesInstanceUID
                self.completion_flag = dataset.CompletionFlag
                self.verification_flag = dataset.VerificationFlag
                self.type = dataset.ContentSequence[0].ValueType
                # Sometimes they're a zip file or something and that's a big fat TODO
                self.json_string = dataset.to_json()


    def __str__(self):
        return (
            'REPORT: ' + '\n' +
            'patient_name     : ' + str(self.patient_name) + '\n' +
            'patient_id       : ' + str(self.patient_id) + '\n' +
            'patient_dob      : ' + str(self.patient_dob) + '\n' +
            'patient_sex      : ' + str(self.patient_sex)+ '\n' +
            'study_uid        : ' + str(self.study_uid)+ '\n' +
            'series_uid       : ' + str(self.series_uid)+ '\n' +
            'completion_flag  : ' + str(self.completion_flag)+ '\n' +
            'verification_flag: ' + str(self.verification_flag)+ '\n' +
            'type             : ' + str(self.type)+ '\n' +
            'text             : ' + str(self.text)+ '\n' +
            'json_string      : ' + str(self.json_string))

    @staticmethod
    def parse_result(result):
        ''' Parse a pynetdicom result and return a report '''
        report = Report()
        for field in result.split('\n'):
            if field.startswith('(0040, a040)'):
                report.type = field.split(': ')[-1].strip("'").strip('"')
            elif field.startswith('(0040, a160)'):
                report.text = field.split(': ')[-1].strip("'").strip('"')
            elif field.startswith('(0040, a491)'):
                report.series_date = field.split(': ')[-1].strip("'").strip('"')
        return report
