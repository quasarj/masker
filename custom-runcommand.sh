#!/bin/bash

TOP=50
BOTTOM=155
LEFT=40
RIGHT=180

docker run \
	-it \
	--rm \
	-v /home/quasar/projects/chris-masker/masker/src:/app \
	-v /home/quasar/projects/chris-masker/masker/../data:/data \
	-v /home/quasar/projects/chris-masker/masker/output:/output \
	masker:latest \
	-i /data/test.nii \
	-o /output \
	--facewindow $TOP $BOTTOM $LEFT $RIGHT
	#--renderonly

	#-i /data/example_data_cpw/nifti_input/t1_mpr_sag_iso.nii.gz \
	# -i /data/CT_for_Bill/16486/input/NIFTI/16486.nii.gz \
