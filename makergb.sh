#!/bin/bash

rm -f tmp/*
./posda-get-series-as-nifti $1 || exit
f=$(ls tmp/*.nii | grep -vi tilt)
#miview "$f"
cp "$f" rgb/$1.nii
