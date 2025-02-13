#!/bin/bash

# activate conda environment first with conda activate tile_stitcher

# set input/output directories
INPUT_DIR="/media/nas1/Python/stitching/stitch_files/"
OUTPUT_DIR="/media/nas1/Python/stitching/fused/"

# get total number of tile sets
N=$(python stitcher_par_linux.py --filepath "$INPUT_DIR" --savepath "$OUTPUT_DIR" --count --tile-index 0)
N=$((N-1))

for i in $(seq 0 $N); do
	echo "Processing tile set $i"
	python stitcher_par_linux.py --filepath "$INPUT_DIR" --savepath "$OUTPUT_DIR" --tile-index $i
done

# to run in parallel
# seq 0 $N | parallel -j 4 python stitcher_par_linux.py --filepath "$INPUT_DIR" --savepath "$OUTPUT_DIR" --tile-index {}