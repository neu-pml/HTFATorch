#!/bin/sh

for file in $1/*.nii ##path to relevant dataset group
do
  fslmaths "$file" -Tmean -bin "${file}_mean"
done

fslmerge -t $1/allmeanmasks4d $1/*.nii.gz
fslmaths $1/allmeanmasks4d -Tmean $1/propDatavox3d
fslmaths $1/propDatavox3d -thr 1 $1/wholebrain
