#!/bin/sh

for file in $1/*.nii.gz ##path to relevant dataset group
do
  base=`basename ${file} .nii.gz`.nii
  gunzip -c ${file} > ${base}
  fslmaths "${base}" -Tmean -bin "${base}_mean"
done

fslmerge -t allmeanmasks4d *.nii.gz
fslmaths allmeanmasks4d -Tmean propDatavox3d
fslmaths propDatavox3d -thr 1 wholebrain
