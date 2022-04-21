#!/bin/bash

rm -f tmp/*
rm -f output/*
./posda-get-series-as-nifti $1 || exit
f=$(ls tmp/*.nii | grep -vi tilt)
#miview "$f"
cp "$f" ../data/test.nii
./runcommand.sh
feh output/

