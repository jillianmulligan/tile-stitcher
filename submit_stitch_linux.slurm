#!/bin/bash
#SBATCH --job-name=tile_stitch
#SBATCH --output=stitch_%A_%a.out
#SBATCH --error=stitch_%A_%a.err

# Load conda
source /opt/conda/etc/profile.d/conda.sh

# Activate the environment
conda activate tile_stitcher

# First get the number of sets
N=$(python stitcher_par.py --filepath "/path/to/Segmentations/Tile_stitching_test/Tiles" --savepath "/path/to/Segmentations/Tile_stitching_test/Tiles/fused" --count --tile-index 0)
N=$((N-1))  # Subtract 1 since array jobs are 0-based

# Submit the array job with the calculated range
sbatch --array=0-${N} <<EOF
#!/bin/bash
#SBATCH --job-name=tile_stitch
#SBATCH --output=stitch_%A_%a.out
#SBATCH --error=stitch_%A_%a.err

# Run the script
python stitcher_par.py --filepath "/path/to/Segmentations/Tile_stitching_test/Tiles" --savepath "/path/to/Segmentations/Tile_stitching_test/Tiles/fused" --tile-index \${SLURM_ARRAY_TASK_ID}
EOF

# Deactivate the environment
conda deactivate 