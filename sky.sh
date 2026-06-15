#!/bin/sh

# sky.dat, sky90_region.dat, and sky90_boundary.dat are now produced
# directly by sky_map.c after the MCMC. This script only renders them.
if [ "$#" -eq 2 ]; then
    tag=$2
else
    tag=$1
fi

if [ -z "$tag" ]; then
    tag=0
fi

rm -f sky.fits

./write_fits sky.dat

python makesky.py

foo=$(printf "%d.fits" "$tag")

mv sky.fits $foo

foo=$(printf "sky_%d.png" "$tag")

cp sky.png $foo

rm -f sky.fits

./write_fits sky90_boundary.dat

python makesky_boundary.py

foo=$(printf "sky90_boundary_%d.fits" "$tag")

mv sky.fits $foo

foo=$(printf "sky90_boundary_%d.png" "$tag")

cp sky90_boundary.png $foo

foo=$(printf "sky90_boundary_check_%d.png" "$tag")

cp sky90_boundary_check.png $foo

foo=$(printf "sky90_boundary_check_zoom_%d.png" "$tag")

cp sky90_boundary_check_zoom.png $foo

open sky.png 
