
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

## Local imports
#from masker.vtk_volume_rendering import * # use when installed as a package
from vtk_volume_rendering import getnumpyrender

## Set up args
#fileName = "D:/UAMS/Project_defacer/testdata/cpw/nifti/t1_mpr_sag_iso.nii.gz" # me
fileName = "D:/UAMS/Project_defacer/testdata/A2/T2.nii.gz" # subject A1
outputDir = "D:/UAMS/Project_defacer/masker_test_output"
modality = "T1"
azimuth = 90
rolls = 0 # this just spins the image around: IN 2D
minMaskThickness=5
maxMaskThickness=15

## Load nifti file
nibfile = nib.load(fileName)
narr = nibfile.get_fdata()
nib.aff2axcodes(nibfile.affine)
narr.shape

#def getnumpyrender(fileName,viewup,azimuth,roll,voxeldims,modality):
#fileName
#viewup # [0,0,-1]
#azimuth # this is rotating the camera around the center
#roll # this is camera roll
#voxeldims
#modality
#arr=getnumpyrender(fileName,[0,0,-1],90,90,narr.shape,modality) # CORRECT FOR MY HEAD EXAMPLE; face on
arr=getnumpyrender(fileName,[0,0,-1],0,90,narr.shape,modality) # T2; face on
arr=np.rot90(arr,k=3)

## Create and save a PNG fo the volume rendering 
im=Image.fromarray(arr)
im.save(outputDir+"/orientationtest.png")
















## Define "correct" orientation; this is the orientation that all code below is set up to process
## After masking the image, we will revert back to the original orientation
LPSorient=[[ 1., -1.],[ 2., -1.],[ 0.,  1.]]

## Get orientation from input file
ornt=nib.orientations.io_orientation(nibfile.affine)
## Calculate required transform and apply reorientation to input file to make is LPS
reorienttoLPS=nib.orientations.ornt_transform(ornt,LPSorient)
narr=nib.orientations.apply_orientation(narr,reorienttoLPS)

#newaffine=nib.orientations.inv_ornt_aff(ornt, narr.shape)

## Save file
#pair_img = nib.Nifti1Pair(narr,newaffine,header=nibfile.header)
#outputfile = outputDir+"/reoriented.nii.gz"
#nib.save(pair_img, outputfile)


## Calculate required transform and apply reorientation to input file to revert it to original orientation
#reorienttooriginal=nib.orientations.ornt_transform(LPSorient,ornt)
#narr=nib.orientations.apply_orientation(narr,reorienttooriginal)

## Save file
pair_img = nib.Nifti1Pair(narr,nibfile.affine,header=nibfile.header)
outputfile = outputDir+"/reoriented.nii.gz"
nib.save(pair_img, outputfile)

############

#cpwornt=nib.orientations.io_orientation(nibfile.affine)
orntrans=nib.orientations.ornt_transform(ornt,cpwornt)
x=nib.orientations.apply_orientation(narr,orntrans)
#
pair_img = nib.Nifti1Pair(x,nibfile.affine,header=nibfile.header)
outputfile = outputDir+"/reoriented.nii.gz"
nib.save(pair_img, outputfile)


# CPW: ('P', 'I', 'R')
# A2 T1:  ('P', 'I', 'R')
# A2 T2: ('L', 'P', 'S')
# CT:  ('L', 'P', 'S')

## Create a volume rendering of the input file
#logging.info("Creating volume rendering of input")
#arr=getnumpyrender(fileName,[0,0,-1],azimuth,90,narr.shape,modality) # CORRECT FOR MY HEAD EXAMPLE; face on
#arr=np.rot90(arr,k=rolls)


## Create and save a PNG fo the volume rendering 
#logging.info("Writing image of volume rendering of input")
#im=Image.fromarray(arr)
#im.save(outputDir+"/volumerendering_b4_"+str(azimuth)+"_roll"+str(rolls)+".png")


import vtk
## Set up reader; can be DICOM or NIFTI
reader = vtk.vtkNIFTIImageReader() # NIFTI
reader.SetFileName(fileName)
reader.GetFileName()

reader.SetFileName("D:/UAMS/Project_defacer/testdata/cpw/nifti/t1_mpr_sag_iso.nii.gz")
reader.GetFileName()
reader.Update()
print(reader.GetNIFTIHeader())
print(reader.GetQFormMatrix())
#z=reader.GetQFormMatrix()

#reader.CheckNIFTIVersion()
#reader.RequestInformation()


print(h)
#cpwDirection=reader.GetDataDirection() # SetDataDirection # (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
T2Direction=reader.GetDataDirection() # SetDataDirection
z.__class__

str(z)

