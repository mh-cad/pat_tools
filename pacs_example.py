# You can replace the values in this file and take 'er for a test drive

from pattools.pacs import *

# These are the settings for your remote SCP, very important.
# Depending on your PACS settings you may need to have your local aet registered with your PACS
# That said, we're just using C_GET and C_FIND requests so maybe not.
remote_scp = ScpSettings('REMOTESCP', 'remote.host', 104, local_aet='ANYSCP')
# ScpSettings take remote AE Title, remote hostname, remote port, local AE title (optional)

# Create a patient instance based on the patient id
pat = Patient('1234567', remote_scp)
print(pat)
print('******************************')

# Explore the studies
studies = pat.find_studies()
for study in studies:
    print(str(study))
    print('----------')
    print(str(study.get_report()))
    print('----------')

# Now we'll try saving some images (nifti may fail if it's not a brain volume).
studies[0].find_series()[0].save_dicom('./testdicom/')
studies[0].find_series()[0].save_nifti('./test.nii.gz')
