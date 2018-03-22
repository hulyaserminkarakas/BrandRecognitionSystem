#!/bin/bash

IMAGES="/home/alperfiratkaya/PycharmProjects/Final/train_greyscale/*.bmp"
for file in $IMAGES
do
	echo "$file"
	convert $file -resize 16x16! -gravity center $file

done
