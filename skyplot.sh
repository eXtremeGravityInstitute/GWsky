#!/bin/sh

# Render the sky map and 90% boundary map for one catalog source.
#
# Usage:
#   ./skyplot.sh SOURCE_INDEX
#
# Inputs are read from:
#   skymaps/sky_SOURCE_INDEX.dat
#   skymaps/sky90_boundary_SOURCE_INDEX.dat
#   skymaps/sky90_region_SOURCE_INDEX.dat  (optional, used for boundary checks)
#
# Outputs are written to:
#   skymaps/sky_SOURCE_INDEX.fits
#   skymaps/sky_SOURCE_INDEX.png
#   skymaps/sky90_boundary_SOURCE_INDEX.fits
#   skymaps/sky90_boundary_SOURCE_INDEX.png
#   skymaps/sky90_boundary_check_SOURCE_INDEX.png
#   skymaps/sky90_boundary_check_zoom_SOURCE_INDEX.png

set -eu

if [ "$#" -ne 1 ]; then
    echo "usage: ./skyplot.sh SOURCE_INDEX" >&2
    exit 1
fi

idx=$1

case "$idx" in
    ''|*[!0-9]*)
        echo "SOURCE_INDEX must be a positive integer" >&2
        exit 1
        ;;
esac

if [ "$idx" -lt 1 ]; then
    echo "SOURCE_INDEX must be >= 1" >&2
    exit 1
fi

script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
map_dir="$script_dir/skymaps"

sky_dat="$map_dir/sky_${idx}.dat"
boundary_dat="$map_dir/sky90_boundary_${idx}.dat"
region_dat="$map_dir/sky90_region_${idx}.dat"

if [ ! -f "$sky_dat" ]; then
    echo "Missing sky map file: $sky_dat" >&2
    exit 1
fi

if [ ! -f "$boundary_dat" ]; then
    echo "Missing boundary map file: $boundary_dat" >&2
    exit 1
fi

if [ ! -x "$script_dir/write_fits" ]; then
    echo "Missing executable write_fits in $script_dir" >&2
    exit 1
fi

tmpdir=$(mktemp -d "${TMPDIR:-/tmp}/skyplot.XXXXXX")
trap 'rm -rf "$tmpdir"' EXIT HUP INT TERM

mplconfig="${TMPDIR:-/tmp}/sky_map_matplotlib"
mkdir -p "$mplconfig"
export MPLCONFIGDIR="$mplconfig"

cp "$sky_dat" "$tmpdir/sky.dat"
cp "$boundary_dat" "$tmpdir/sky90_boundary.dat"
if [ -f "$region_dat" ]; then
    cp "$region_dat" "$tmpdir/sky90_region.dat"
fi

(
    cd "$tmpdir"

    "$script_dir/write_fits" sky.dat
    python "$script_dir/makesky.py"
    mv sky.fits "$map_dir/sky_${idx}.fits"
    cp sky.png "$map_dir/sky_${idx}.png"

    "$script_dir/write_fits" sky90_boundary.dat
    python "$script_dir/makesky_boundary.py"
    mv sky.fits "$map_dir/sky90_boundary_${idx}.fits"
    cp sky90_boundary.png "$map_dir/sky90_boundary_${idx}.png"

    if [ -f sky90_boundary_check.png ]; then
        cp sky90_boundary_check.png "$map_dir/sky90_boundary_check_${idx}.png"
    fi

    if [ -f sky90_boundary_check_zoom.png ]; then
        cp sky90_boundary_check_zoom.png "$map_dir/sky90_boundary_check_zoom_${idx}.png"
    fi
)

echo "Wrote $map_dir/sky_${idx}.png"
echo "Wrote $map_dir/sky90_boundary_${idx}.png"
