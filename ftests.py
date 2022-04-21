#!/usr/bin/env python3

import sys
import nibabel

file = sys.argv[1]

nifti_file = nibabel.load(file)


header_lines = [
    "datatype",
    "bitpix",
    "scl_slope",
    "scl_inter"
]
for h in header_lines:
    print(h, nifti_file.header[h])

data = nifti_file.get_fdata()
print(data[:2])
