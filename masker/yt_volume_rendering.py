

## Import packages
import sys
import nibabel as nib
import numpy as np
from PIL import Image, ImageDraw
import face_recognition
from sklearn.cluster import KMeans
from scipy import spatial
from scipy.signal import savgol_filter
import argparse
import logging
import datetime

##
#import matplotlib 
import yt
## NOTES ON yt package;
#Ah, this is a recent issue -- you can a) pull from the most recent yt,
#which no longer registers its own cubehelix, b) downgrade matplotlib ## USE matplotlib==3.0.3
#to pre-3.4, or c) (maybe the easiest?) edit
#~/anaconda3/lib/python3.7/site-packages/yt/visualization/color_maps.py
#and change line 53 to be:
#except (AttributeError, ValueError):
#instead of just AttributeError.

## Set up args
fileName = "D:/UAMS/Project_defacer/testdata/cpw/nifti/t1_mpr_sag_iso.nii.gz" # me
#fileName = "D:/UAMS/Project_defacer/testdata/A5/nifti/T1.nii.gz" # subject A5
#fileName = "D:/UAMS/Project_defacer/testdata/A8/A8T1.nii.gz" # subject A8
#fileName = "D:/UAMS/Project_defacer/testdata/A2/T1.nii.gz" # subject A1
#fileName = "D:/UAMS/Project_defacer/testdata/A2/CT1.nii.gz" # subject A1
#fileName = "D:/UAMS/Project_defacer/testdata/A2/T2.nii.gz" # subject A1
#fileName = "D:/UAMS/Project_defacer/testdata/A2/FLAIR.nii.gz" # subject A1
outputDir = "D:/UAMS/Project_defacer/masker_test_output"
modality = "T1"
azimuth = 90
rolls = 0
minMaskThickness=5
maxMaskThickness=15

## Load nifti file
logging.info("Loading NIFTI input file")
nibfile = nib.load(fileName)
narr = nibfile.get_fdata()


ds = yt.load("IsolatedGalaxy/galaxy0030/galaxy0030")

arr = np.random.random(size=(64,64,64))

ds = yt.load_uniform_grid(data, arr.shape, length_unit="Mpc", bbox=bbox, nprocs=64)


sc = yt.create_scene(narr)