#!/usr/bin/env python3

# Copyright 2021 CP Wardell
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

## Title: Masker
## Author: CP Wardell
## Description: Anonymizes medical imaging data by detecting and obscuring faces
## Takes a NIFTI file, creates a volume rendering and detects faces 
## Designed and tested for MRI, will be extended to support CT data

########################################
## Things to make and do - pseudocode ##
########################################

## Read in file - DONE
## Save volume rendering image for QC - DONE
## Find face - DONE
## Save face location image for QC - DONE
## Add mask - DONE
## Save volume rendering image for QC - DONE
## Save masked nifti - DONE

########################################
########################################
########################################

## Import packages
import nibabel as nib
import numpy as np
from PIL import Image, ImageDraw
import face_recognition
from sklearn.cluster import KMeans
from scipy import spatial
from scipy.signal import savgol_filter

## For logging and debugging
import datetime

## Local imports
#from masker.vtk_volume_rendering import * # use when installed as a package
from vtk_volume_rendering import getnumpyrender

## Set up args
#fileName = "D:/UAMS/Project_defacer/testdata/cpw/nifti/t1_mpr_sag_iso.nii.gz" # me
#fileName = "D:/UAMS/Project_defacer/testdata/A5/nifti/T1.nii.gz" # subject A5
fileName = "D:/UAMS/Project_defacer/testdata/A8/A8T1.nii.gz" # subject A8
outputDir = "D:/UAMS/Project_defacer/masker_test_output"
modality = "T1"
azimuth = 90
rolls = 0

## Load nifti file
nibfile = nib.load(fileName)
narr = nibfile.get_fdata()

## Create a volume rendering of the input file
arr=getnumpyrender(fileName,[0,0,-1],azimuth,90,narr.shape,modality) # CORRECT FOR MY HEAD EXAMPLE; face on
arr=np.rot90(arr,k=rolls)

## Create and save a PNG fo the volume rendering 
im=Image.fromarray(arr)
im.save(outputDir+"/volumerendering_b4_"+str(azimuth)+"_roll"+str(rolls)+".png")
#im.show() # uncomment this for the file to open in Explorer

## Pass in volume rendering to facial recognition package and return face bounding box
face_locations=face_recognition.face_locations(arr, number_of_times_to_upsample=0, model="cnn") # using deep learning

## If no faces are found, inform the user and exit gracefully 
if(len(face_locations)==0):
    print("No faces found in the input data")
    sys.exit()
else:
    top, right, bottom, left = face_locations[0]

# Draw a box around the detected face and output an image
draw = ImageDraw.Draw(im)
draw.rectangle(((left, top), (right, bottom)), outline=(255, 0, 0))
im.save(outputDir+"/volumerendering_b4_box_"+str(azimuth)+"_roll"+str(rolls)+".png")

## Generate kmeans clustering of numpy image array; note we reshape the array
## This can't be sped up by using only unique values; it would shift the centers of the clusters
#startkmeans=datetime.datetime.now()
kmeans = KMeans(n_clusters=2,random_state=666,algorithm="full").fit(narr.reshape(-1,1))
#endkmeans=datetime.datetime.now()
#print(kmeans.n_iter_)
#print(kmeans.cluster_centers_)
#print("Total time taken: "+str(endkmeans-startkmeans))


## Determine which label is background; this is assigned to the cluster with the lower center
if( float(kmeans.cluster_centers_[[0]]) < float(kmeans.cluster_centers_[[1]]) ):
    background,subject = 0,1
else:
    background,subject = 1,0

## Calculate median value of all subject voxels so we can mask with similar noise later
subjectmean=np.mean(narr.reshape(-1,1)[np.where(kmeans.labels_==subject)])
subjectstd=np.std(narr.reshape(-1,1)[np.where(kmeans.labels_==subject)])

## Reshape the kmeans labels so that we have a binary arrary
## We can iterate through this to determine where the surface of the face is
bk=kmeans.labels_.reshape(narr.shape)

## Output binary nifti file if desired
pair_img = nib.Nifti1Pair(bk,nibfile.affine,header=nibfile.header)
nib.save(pair_img, outputDir+"/bk.nii.gz")

## Object to store the first non-zero voxel; this is the surface of the face
for i in range(0,bk.shape[2]): # from left to right
    surfacevoxels=[]
    for j in range(0,bk.shape[1]): # from top to bottom
        surfacevoxels.append(bk.shape[1]-1)
        nonzeros=[k for k, x in enumerate(bk[:,j,i]) if x == subject ] # these are nonzero stacks
        if(len(nonzeros)!=0):
            firstnonzero=nonzeros[0]
            surfacevoxels[j]=firstnonzero
    ## Smooth the binary volume, then go back and backfill stacks of voxels
    smoothedvoxels = savgol_filter(surfacevoxels, 41, 2) # window size 41, polynomial order 2
    smoothedvoxels=np.clip(np.round(smoothedvoxels,0),0,bk.shape[1])
    for j in range(0,bk.shape[1]): # from top to bottom
        bk[range(int(smoothedvoxels[j]),int(round(bk.shape[1]*0.6,0))),j,i]=subject # fill in voxels from back to front # GO 0.6 of the way down

## Output binary nifti file if desired
pair_img = nib.Nifti1Pair(bk,nibfile.affine,header=nibfile.header)
nib.save(pair_img, outputDir+"/bk.smoothed.nii.gz")

## Create fake voronoi segmentation using kmeans on random data
## Ranges between 0 and the dimensions of the real matrix so that the points can be mapped to one another
vdatax=np.random.uniform(0,narr.shape[2],10000)
vdatay=np.random.uniform(0,narr.shape[1],10000)
vdata=np.column_stack((vdatax,vdatay))
## Number of clusters; 300 is ok for head, 600 for body, but we use the max dimensions of the input... maybe make this an option for users?
clusters=max(narr.shape)
vkmeans = KMeans(n_clusters=clusters,random_state=0).fit(vdata)
vlabels=vkmeans.labels_

## Generate matrix of all coordinates in a grid
firstcol=np.tile(range(0,narr.shape[2]),narr.shape[1])
secondcol=np.repeat(range(0,narr.shape[1]),narr.shape[2])
matrixcoords=np.column_stack((firstcol,secondcol))

## Use nearest-neighbour to determine which kmeans cluster each coordinate belongs to
def nearest_neighbour(points_a, points_b):
    tree = spatial.cKDTree(points_b)
    return tree.query(points_a)[1]
nearestindex=nearest_neighbour(matrixcoords,vdata)

## Set voronoi cluster height and depth;
## Determine highest point in each voronoi patch; 
## One for every cluster, set to last voxel of stack
localpeak=np.ones(clusters,int)*narr.shape[0]
## Iterate through the binary kmeans array using the face window as the edges
## We must consider viewup, azimuth and roll to determine the direction we iterate from
for i in range(0,narr.shape[2]): # from left to right
#for i in range(adj_face_locations[3],adj_face_locations[1]): # from left to right
    for j in range(0,narr.shape[1]): # from top to bottom
    #for j in range(adj_face_locations[0],adj_face_locations[2]): # from top to bottom
        #for k in range(0,narr.shape[2]): # from front to back
        nonzeros=[k for k, x in enumerate(bk[:,j,i]) if x == subject ]
        if(len(nonzeros)!=0):
            xvalues=np.where(matrixcoords[:,0]==i)
            yvalues=np.where(matrixcoords[:,1]==j)
            x=set(list(xvalues[0]))
            y=set(list(yvalues[0]))
            index=x.intersection(y)
            thiscluster=vlabels[nearestindex[list(index)[0]]] # sets surface voxel to same value as cluster
            firstnonzero=nonzeros[0]
            if(firstnonzero<localpeak[thiscluster]):
                localpeak[thiscluster]=firstnonzero

## This is how far the clusters will be extended above the current surface
## 1 to 5 voxels above the maximum height of any cluster
clusterdepth=np.random.randint(1,6,clusters)
for i in range(0,len(localpeak)):
    localpeak[i]=max(0,localpeak[i]-clusterdepth[i])

## Correct face locations; pixel coordinates for arr may not be the same as voxel dimensions
## for narr due to non-isometric voxels
xscale=narr.shape[2]/arr.shape[1]
yscale=narr.shape[1]/arr.shape[0]

adj_face_locations=[0,0,0,0]
adj_face_locations[1]=int(face_locations[0][1]*xscale) # right
adj_face_locations[3]=int(face_locations[0][3]*xscale) # left
adj_face_locations[0]=int(face_locations[0][0]*yscale) # up
adj_face_locations[2]=int(face_locations[0][2]*yscale) # down

## Do not allow the mask to be deeper than the width of the face
#maxdepth=abs(adj_face_locations[1]-adj_face_locations[3])

## Iterate through the binary kmeans array using the face window as the edges
## We must consider viewup, azimuth and roll to determine the direction we iterate from
#for i in range(0,narr.shape[2]): # from left to right
#for i in range(face_locations[0][3],face_locations[0][1]): # from left to right
for i in range(adj_face_locations[3],adj_face_locations[1]): # from left to right
#    for j in range(0,narr.shape[1]): # from top to bottom
     #for j in range(face_locations[0][0],face_locations[0][2]): # from top to bottom
     for j in range(adj_face_locations[0],adj_face_locations[2]): # from top to bottom
        #for k in range(0,narr.shape[2]): # from front to back
        nonzeros=[k for k, x in enumerate(bk[:,j,i]) if x == subject ]
        if(len(nonzeros)!=0):
            xvalues=np.where(matrixcoords[:,0]==i)
            yvalues=np.where(matrixcoords[:,1]==j)
            x=set(list(xvalues[0]))
            y=set(list(yvalues[0]))
            index=x.intersection(y)
            thiscluster=vlabels[nearestindex[list(index)[0]]] # the voronoi cluster we need
            #bk[nonzeros[0],j,i]=vlabels[nearestindex[list(index)[0]]] # sets surface voxel to same value as cluster
            ## Blocks may not be longer than 20% of the total z depth of the volume; set them to 15%
            if(abs(localpeak[thiscluster]-nonzeros[0]) > narr.shape[0]*0.20):
                nonzeros[0] = localpeak[thiscluster]+round(narr.shape[0]*0.15)
            fillme=range(localpeak[thiscluster],nonzeros[0]) 
            randvalues=np.random.normal(subjectmean,subjectstd,len(fillme)) # random values based on non-background voxels
            #bk[fillme,j,i] = randvalues # sets range of voxels to random values # using BINARY MAP
            narr[fillme,j,i] = randvalues # sets range of voxels to random values absed # using original data
            
## Final clean up; set any negative voxel to 0 and convert all values back to ints
narr[narr<0]=0
narr=narr.astype(int)

#### Here we reorienate the narr data as necessary by rotating it with np.rot90 etc
## if necessary using the azimuth and rolls variables; put it back the way it was
#print(azimuth)
#print(rolls)

## Correct for azimuth
# if azimuth is 90, do nothing
#if(azimuth==0):
#    #narr=np.rot90(narr,k=1,axes=(1,2)) # mask on butt
#    #narr=np.rot90(narr,k=3,axes=(2,1)) # TRIED
#    narr=np.flip(narr,0) # flip the depth axis
#    narr=np.flip(narr,2) # flip the LR axis
#    narr=np.rot90(narr,k=3,axes=(1,2)) # 

## Correct for rolls
#narr=np.rot90(narr,4-rolls)

## Write results to file
## MUST USE ORIGINAL AFFINE MATRIX FROM INPUT FILE
#pair_img = nib.Nifti1Pair(bk,nibfile.affine,header=nibfile.header)
#nib.save(pair_img, 'bk6.nii.gz')

## Write results to file
## MUST USE ORIGINAL AFFINE MATRIX FROM INPUT FILE
pair_img = nib.Nifti1Pair(narr,nibfile.affine,header=nibfile.header)
outputfile = outputDir+"/masked.nii.gz"
nib.save(pair_img, outputfile)

## Create a volume rendering of the output file
arr=getnumpyrender(outputfile,[0,0,-1],azimuth,90,narr.shape,modality) 
arr=np.rot90(arr,k=rolls)

## Create and save a PNG fo the volume rendering 
im=Image.fromarray(arr)
im.save(outputDir+"/masked"+str(azimuth)+"_roll"+str(rolls)+".png")

exit()
