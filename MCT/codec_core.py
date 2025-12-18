import pydicom
import numpy as np



ds = pydicom.dcmread("/media/user/driver/MMIP/MCT/2_skull_ct/DICOM/I0")
img = ds.pixel_array.astype(np.int16)

print(img.shape, img.dtype)
# e.g. (512, 512), int16