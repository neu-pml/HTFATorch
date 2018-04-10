#!/bin/sh

for file in /home/zulqarnain/Datasets/new_pieman/pieman_data_intact/*.nii ##path to relevant dataset group
do
  fslmaths "$file" -Tmean -bin "${file}_mean"
done

fslmerge -t allmeanmasks4d /home/zulqarnain/Datasets/new_pieman/pieman_data_intact/*.nii.gz
fslmaths allmeanmasks4d -Tmean propDatavox3d
fslmaths propDatavox3d -thr 1 wholebrain
