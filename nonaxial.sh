#!/bin/bash

rm -f tmp/*
rm -f output/*
./posda-get-series-as-nifti $1
f=$(ls tmp/*.nii | grep -vi tilt)
#miview "$f"
cp "$f" ../data/test.nii
./runcommand.sh

cp "$f" "non-axial/$1.nii"
ls "non-axial/$1.nii"
