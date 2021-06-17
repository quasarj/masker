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
## Designed and tested for MRI, extended to support CT data

###########################
## Things to make and do ## 
###########################

## Push to PyPi and Bioconda
## Make Dockerfile

########################################
########################################
########################################

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

def main():

    ## Gather command line args
    ## Create a new argparse class that will print the help message by default
    class MyParser(argparse.ArgumentParser):
        def error(self, message):
            sys.stderr.write('error: %s\n' % message)
            self.print_help()
            sys.exit(2)
    parser=MyParser(description="Masker: Anonymizes medical imaging data by detecting and obscuring faces", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-i", type=str, help="path to input NIFTI file", required=True)
    parser.add_argument("-o", type=str, help="path to input output directory", required=True)
    parser.add_argument("-s", type=int, help="Smoothing window; make larger to make mask interface less smooth", required=False, default=4)
    parser.add_argument("--binary", help="Emit binarized NIFTI files; useful for debugging", required=False, default=False, action='store_true')
    parser.add_argument("--renderonly", help="Produce volume rendering of input file and face detection, then exit without masking", required=False, default=False, action='store_true')
    parser.add_argument("--maskthickness", type=int, help="Minimum and maximum mask depth in voxels", required=False, default=[3,7], nargs=2)
    parser.add_argument("--facewindow", type=int, help="Manual override to create masking window over face (in mm); top, bottom, left, right", required=False, default=[0,0,0,0], nargs=4)
    args=parser.parse_args()

    ## Turn arguments into a nice string for printing
    printargs=str(sys.argv)
    printargs=printargs.replace(",","")
    printargs=printargs.replace("'","")
    printargs=printargs.replace("[","")
    printargs=printargs.replace("]","")

    ## Assign command line args to more friendly variable names
    fileName = args.i
    outputDir = args.o
    smoothness = args.s
    minMaskThickness = args.maskthickness[0]
    maxMaskThickness = args.maskthickness[1]
    facetop = args.facewindow[0]
    facebottom = args.facewindow[1]
    faceleft = args.facewindow[2]
    faceright = args.facewindow[3]

    ## Set up logging
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s',datefmt='%Y-%m-%d %H:%M:%S',filename=outputDir+'/log.txt',filemode='w')
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s',datefmt='%Y-%m-%d %H:%M:%S')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)

    ## Report start-up conditions and variables to user
    starttime=datetime.datetime.now()
    logging.info("Masker was invoked using this command: "+printargs)
    logging.info("Start time is: "+str(starttime))
    logging.info("Input file is: "+str(fileName))
    logging.info("Output directory is: "+str(outputDir))
    logging.info("Mask window smoothing is: "+str(smoothness))
    logging.info("Minimum mask thickness in voxels is: "+str(minMaskThickness))
    logging.info("Maximum mask thickness in voxels is: "+str(maxMaskThickness))
        
    ## Load nifti file
    logging.info("Loading NIFTI input file")
    nibfile = nib.load(fileName)
    narr = nibfile.get_fdata()

    ## Get orientation of data
    orientation = nib.aff2axcodes(nibfile.affine)
    ostring=''.join(orientation)
    logging.info("Detected image orientation: "+ostring)

    ## Left-right axis
    LR = ostring.find("R")
    if LR == -1:
        LR = ostring.find("L")
    ## Superior-inferior axis
    SI = ostring.find("I")
    if SI == -1:
        SI = ostring.find("S")
    ## Posterior-anterior axis
    PA = ostring.find("P")
    if PA == -1:
        PA = ostring.find("A")

    ## Find real size of volume in mm
    vsizes = nib.affines.voxel_sizes(nibfile.affine)
    vsizeLR = vsizes[LR]
    vsizeSI = vsizes[SI]
    vsizePA = vsizes[PA]

    ## Report image dimensions:
    logging.info("LR dimension: " + str(narr.shape[LR]) + " " + str(np.round(vsizeLR,1)) + " mm voxels, total width " + str(np.round(vsizeLR*narr.shape[LR],1)) + " mm")
    logging.info("SI dimension: " + str(narr.shape[SI]) + " " + str(np.round(vsizeSI,1)) + " mm voxels, total height " + str(np.round(vsizeSI*narr.shape[SI],1)) + " mm")
    logging.info("PA dimension: " + str(narr.shape[PA]) + " " + str(np.round(vsizePA,1)) + " mm voxels, total depth " + str(np.round(vsizePA*narr.shape[PA],1)) + " mm")

    ## Check that facewindow (if set) doesn't exceed maximum dimensions
    if(args.facewindow != [0,0,0,0]):
        if(args.facewindow[0] < 0):
            logging.info("Top component of facewindow is outside plotting area; exiting")
            sys.exit()
        if(args.facewindow[2] < 0):
            logging.info("Left component of facewindow is outside plotting area; exiting")
            sys.exit()
        if(args.facewindow[1] > (np.round(vsizeSI*narr.shape[SI],1))):
            logging.info("Bottom component of facewindow is outside plotting area; exiting")
            sys.exit()
        if(args.facewindow[3] > (np.round(vsizeLR*narr.shape[LR],1))):
            logging.info("Right component of facewindow is outside plotting area; exiting")
            sys.exit()

    ## Create a volume rendering of the input file
    logging.info("Creating volume rendering of input")
    minvoxel=np.min(narr)
    maxvoxel=np.max(narr)
    if ostring == "PIR":
        arr=getnumpyrender(fileName,[0,0,-1],90,90,narr.shape,minvoxel,maxvoxel)
    if ostring == "LPS":
        arr=getnumpyrender(fileName,[0,0,-1],0,90,narr.shape,minvoxel,maxvoxel)
        arr=np.rot90(arr,k=3)

    ## Create and save a PNG fo the volume rendering 
    logging.info("Writing image of volume rendering of input")
    im=Image.fromarray(arr)
    im.save(outputDir+"/vr_before.png")

    ## Pass in volume rendering to facial recognition package and return face bounding box
    face_locations=face_recognition.face_locations(arr, number_of_times_to_upsample=0, model="cnn") # using deep learning

    ## If no faces are found and no --facewindow parameter was set, inform the user and exit gracefully 
    if(len(face_locations)==0 and args.facewindow == [0,0,0,0]):
        logging.info("No faces found in the input data; exiting")
        sys.exit()
    ## If faces are detected and no --facewindow parameter was set, set face window to detected positions
    if(len(face_locations)!=0 and args.facewindow == [0,0,0,0]):
        facetop, faceright, facebottom, faceleft = face_locations[0]
        logging.info("Face located at top, right, bottom, left: "+ str(face_locations[0]))

    ## Draw a box around the detected face and output an image
    draw = ImageDraw.Draw(im)
    draw.rectangle(((faceleft, facetop), (faceright, facebottom)), outline=(255, 0, 0))
    logging.info("Writing image of face location in volume rendering of input")
    im.save(outputDir+"/vr_facebox.png")

    ## Exit if user only wants renders and face detection
    if(args.renderonly):
        logging.info("--renderonly selected, exiting")
        sys.exit()

    ## Correct face locations; pixel coordinates for arr will not be the same as voxel dimensions
    ## unless voxels are exactly 1mm
    logging.info("Converting face image coordinates to NIFTI image coordinates")
    xscale=narr.shape[LR]/arr.shape[1]
    yscale=narr.shape[SI]/arr.shape[0]

    adj_face_locations=[0,0,0,0]
    if("R" in ostring):
        adj_face_locations[3]=narr.shape[LR]-int(faceright*xscale) # right
        adj_face_locations[1]=narr.shape[LR]-int(faceleft*xscale) # left
    if("L" in ostring):
        adj_face_locations[1]=int(faceright*xscale) # right
        adj_face_locations[3]=int(faceleft*xscale) # left
    
    if("I" in ostring):
        adj_face_locations[0]=int(facetop*yscale) # top
        adj_face_locations[2]=int(facebottom*yscale) # bottom
    if("S" in ostring):
        adj_face_locations[2]=narr.shape[SI]-int(facetop*yscale) # top
        adj_face_locations[0]=narr.shape[SI]-int(facebottom*yscale) # bottom

  
    ## Generate kmeans clustering of numpy image array; note we reshape the array to form a 1D array
    ## This can't be sped up by using only unique values; it would shift the centers of the clusters
    logging.info("Binarizing input data")
    kmeans = KMeans(n_clusters=2,random_state=666,algorithm="full").fit(narr.reshape(-1,1))

    ## Determine which label is background; this is assigned to the cluster with the lower center
    if( float(kmeans.cluster_centers_[[0]]) < float(kmeans.cluster_centers_[[1]]) ):
        background,subject = 0,1
    else:
        background,subject = 1,0

    ## Calculate median value of all subject voxels so we can mask with similar noise later
    logging.info("Calculating mean and SD of subject voxels")
    subjectmean=np.mean(narr.reshape(-1,1)[np.where(kmeans.labels_==subject)])
    subjectstd=np.std(narr.reshape(-1,1)[np.where(kmeans.labels_==subject)])

    ## Reshape the kmeans labels so that we have a binary arrary
    ## We can iterate through this to determine where the surface of the face is
    bk=kmeans.labels_.reshape(narr.shape)

    ## Output binary nifti file if desired
    if(args.binary):
        logging.info("Writing binarized NIFTI file")
        pair_img = nib.Nifti1Pair(bk,nibfile.affine,header=nibfile.header)
        nib.save(pair_img, outputDir+"/binarized.nii.gz")
            
    logging.info("Smoothing surface of binarized data")
    for i in range(0,bk.shape[LR]): # from left to right
        surfacevoxels=[]
        for j in range(0,bk.shape[SI]): # from top to bottom
            surfacevoxels.append(bk.shape[PA]-1)
            slots=[0,0,0]
            slots[LR]=i
            slots[SI]=j
            slots[PA]=slice(bk.shape[PA])
            nonzeros=[k for k, x in enumerate(bk[tuple(slots)]) if x == subject ] # these are nonzero stacks
            if(len(nonzeros)!=0):
                firstnonzero=nonzeros[0]
                surfacevoxels[j]=firstnonzero
        ## Smooth the binary volume, then go back and backfill stacks of voxels
        ## Decide smoothing window size; MUST be odd
        savgolwindow=int(np.round(bk.shape[SI]/smoothness,0))
        if( savgolwindow % 2 == 0):
            savgolwindow = savgolwindow - 1
        smoothedvoxels = savgol_filter(surfacevoxels, savgolwindow, 2) # window size SI/smoothness, polynomial order 2
        
        for j in range(0,bk.shape[SI]): # from top to bottom
            slots=[0,0,0]
            slots[LR]=i
            slots[SI]=j
            slots[PA]=range(int(smoothedvoxels[j]),int(round(bk.shape[PA]*0.6,0)))
            bk[tuple(slots)]=subject # fill in voxels from back to front and go 60% of the way down

    ## Output binary nifti file if desired
    if(args.binary):
        logging.info("Writing smoothed binarized NIFTI file")
        pair_img = nib.Nifti1Pair(bk,nibfile.affine,header=nibfile.header)
        nib.save(pair_img, outputDir+"/binarized_smoothed.nii.gz")
        
    ## Create fake voronoi segmentation using kmeans on random data
    ## Ranges between 0 and the dimensions of the real matrix so that the points can be mapped to one another
    logging.info("Creating Voronoi tesselation")
    vdatax=np.random.uniform(0,narr.shape[LR],10000)
    vdatay=np.random.uniform(0,narr.shape[SI],10000)
    vdata=np.column_stack((vdatax,vdatay))
    ## Number of voronoi clusters; set to maximum dimension in the coronal plane
    clusters=np.max(arr.shape)
    logging.info("Voronoi tesselation will use "+str(clusters)+" clusters")
    vkmeans = KMeans(n_clusters=clusters,random_state=0).fit(vdata)
    vlabels=vkmeans.labels_

    ## Generate matrix of all coordinates in a grid
    firstcol=np.tile(range(0,narr.shape[LR]),narr.shape[SI])
    secondcol=np.repeat(range(0,narr.shape[SI]),narr.shape[LR])
    matrixcoords=np.column_stack((firstcol,secondcol))

    ## Use nearest-neighbour to determine which kmeans cluster each coordinate belongs to
    def nearest_neighbour(points_a, points_b):
        tree = spatial.cKDTree(points_b)
        return tree.query(points_a)[1]
    nearestindex=nearest_neighbour(matrixcoords,vdata)

    ## Set voronoi cluster height and depth;
    ## Determine highest point in each voronoi patch; 
    ## One for every cluster, set to last voxel of stack
    logging.info("Locating surface of face")
    localpeak=np.ones(clusters,int)*narr.shape[PA]
    ## Iterate through the binary kmeans array using the face window as the edges
    for i in range(adj_face_locations[3],adj_face_locations[1]): # from left to right
        for j in range(adj_face_locations[0],adj_face_locations[2]): # from top to bottom
            slots=[0,0,0]
            slots[LR]=i
            slots[SI]=j
            slots[PA]=slice(bk.shape[PA])
            nonzeros=[k for k, x in enumerate(bk[tuple(slots)]) if x == subject ] # these are nonzero stacks
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
    ## minMaskThickness to maxMaskThickness voxels above the maximum height of any cluster
    logging.info("Assigning mask thickness for each Voronoi cell")
    clusterdepth=np.random.randint(minMaskThickness,maxMaskThickness+1,clusters)
    for i in range(0,len(localpeak)):
        localpeak[i]=max(0,(localpeak[i]-clusterdepth[i]))

    ## Iterate through the binary kmeans array using the face window as the edges
    logging.info("Building mask")
    for i in range(adj_face_locations[3],adj_face_locations[1]): # from left to right
         for j in range(adj_face_locations[0],adj_face_locations[2]): # from top to bottom
            slots=[0,0,0]
            slots[LR]=i
            slots[SI]=j
            slots[PA]=slice(bk.shape[PA])
            nonzeros=[k for k, x in enumerate(bk[tuple(slots)]) if x == subject ] # these are nonzero stacks
            if(len(nonzeros)!=0):
                xvalues=np.where(matrixcoords[:,0]==i)
                yvalues=np.where(matrixcoords[:,1]==j)
                x=set(list(xvalues[0]))
                y=set(list(yvalues[0]))
                index=x.intersection(y)
                thiscluster=vlabels[nearestindex[list(index)[0]]] # the voronoi cluster we need
                ## Blocks may not be longer than 20% of the total z depth of the volume; set them to 15%
                if(abs(localpeak[thiscluster]-nonzeros[0]) > narr.shape[PA]*0.20):
                    nonzeros[0] = localpeak[thiscluster]+round(narr.shape[PA]*0.15)
                fillme=range(localpeak[thiscluster],nonzeros[0]) 
                randvalues=np.random.normal(subjectmean,subjectstd,len(fillme)) # random values based on non-background voxels
                randvalues=list(map(abs,randvalues)) # this may introduce negative values, so we make them all positive
                slots=[0,0,0]
                slots[LR]=i
                slots[SI]=j
                slots[PA]=fillme
                narr[tuple(slots)] = randvalues # sets range of voxels to random values
            
    ## Final clean up; convert all values back to ints
    narr=narr.astype(int)

    ## Write results to file
    ## MUST USE ORIGINAL AFFINE MATRIX FROM INPUT FILE
    logging.info("Writing masked NIFTI file")
    pair_img = nib.Nifti1Pair(narr,nibfile.affine,header=nibfile.header)
    outputfile = outputDir+"/masked.nii.gz"
    nib.save(pair_img, outputfile)

    ## Create a volume rendering of the output file
    if ostring == "PIR":
        arr=getnumpyrender(outputfile,[0,0,-1],90,90,narr.shape,minvoxel,maxvoxel) 
    if ostring == "LPS":
        arr=getnumpyrender(outputfile,[0,0,-1],0,90,narr.shape,minvoxel,maxvoxel)
        arr=np.rot90(arr,k=3)

    ## Create and save a PNG fo the volume rendering 
    logging.info("Creating volume rendering of output")
    im=Image.fromarray(arr)
    logging.info("Writing image of masked volume rendering output")
    im.save(outputDir+"/vr_masked.png")

    endtime=datetime.datetime.now()
    logging.info("End time is: "+str(endtime))
    logging.info("Total time taken: "+str(endtime-starttime))

    exit()


## Execute main method
if __name__ == '__main__':
    main()
