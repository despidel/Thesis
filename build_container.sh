#!/bin/bash
set -e  # Exit immediately if a command fails

# --- DEFAULTS ---
# Default save location if no argument is provided
OUTPUT_DIR="$HOME/neuro_software"
IMAGE_NAME="neuro_suite.sif"
LICENSE_PATH="./license.txt"

# --- ARGUMENT PARSING ---
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --output_dir) OUTPUT_DIR="$2"; shift ;;
        --license_path)    LICENSE_PATH="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

# Ensure output directory exists and get absolute path
mkdir -p "$OUTPUT_DIR"
ABS_OUTPUT_DIR="$(cd "$OUTPUT_DIR" && pwd)"
FINAL_PATH="$ABS_OUTPUT_DIR/$IMAGE_NAME"

# Define paths for intermediate files inside the output directory
TEMP_DOCKERFILE="$ABS_OUTPUT_DIR/Dockerfile"
TEMP_TAR="$ABS_OUTPUT_DIR/neuro_temp.tar"
TEMP_LICENSE="$ABS_OUTPUT_DIR/temp_fs_license.txt"

echo "========================================================"
echo "   Neuroimaging Container Builder (FS + FSL + ANTs)"
echo "========================================================"

# --- WARNING BLOCK ---
echo ""
echo "WARNING: HIGH DISK SPACE & BANDWIDTH USAGE"
echo "Files will be saved to: $OUTPUT_DIR"
echo "Depending on your internet connection, this may take 1 hour+."
echo ""
echo "Output location: $FINAL_PATH"
echo "License source:  $LICENSE_PATH"
echo ""
read -p "Press [Enter] to continue or Ctrl+C to cancel..."
echo ""

# 1. Check for Prerequisites
echo "[1/6] Checking prerequisites..."

if ! command -v docker &> /dev/null; then
    echo "Error: Docker is not installed or running."
    exit 1
fi

if ! command -v apptainer &> /dev/null; then
    echo "Error: Apptainer (Singularity) is not installed."
    exit 1
fi

if [ ! -f "$LICENSE_PATH" ]; then
    echo "Error: License file not found at: $LICENSE_PATH"
    echo "Please download your FreeSurfer license or specify path with --license"
    exit 1
fi

# Prepare build context: Copy license to output dir to be accessible by Docker
cp "$LICENSE_PATH" "$TEMP_LICENSE"

# 2. Generate Dockerfile
echo "[2/6] Generating Dockerfile in output directory..."

# Note: We copy 'temp_fs_license.txt' because we moved it into the build context
docker run --rm repronim/neurodocker:master generate docker \
  --base-image ubuntu:22.04 \
  --pkg-manager apt \
  --freesurfer version=7.4.1 \
  --fsl version=6.0.7.1 \
  --ants version=2.4.3 \
  --copy temp_fs_license.txt /opt/freesurfer/license.txt \
  --env "FS_LICENSE=/opt/freesurfer/license.txt" \
  --env "FSLOUTPUTTYPE=NIFTI_GZ" \
  --yes \
  > "$TEMP_DOCKERFILE"

# 3. Build Docker Image
echo "[3/6] Building intermediate Docker image..."
echo "      (This takes a long time: ~1 hour)"
# We use ABS_OUTPUT_DIR as the build context so it finds the Dockerfile and License
docker build -t neuro_suite_local -f "$TEMP_DOCKERFILE" "$ABS_OUTPUT_DIR"

# 4. Save to Tarball
echo "[4/6] Exporting Docker image to temporary file..."
echo "      (Writing to $TEMP_TAR...)"
docker save neuro_suite_local -o "$TEMP_TAR"

# 5. Convert to Apptainer
echo "[5/6] Converting to Apptainer (.sif)..."
apptainer build "$FINAL_PATH" "docker-archive://$TEMP_TAR"

# 6. Cleanup
echo "[6/6] Cleaning up temporary files..."
rm "$TEMP_DOCKERFILE"
rm "$TEMP_TAR"
rm "$TEMP_LICENSE"
docker rmi neuro_suite_local

echo "========================================================"
echo "SUCCESS!"
echo "Container saved at: $FINAL_PATH"
echo "========================================================"
echo ""
echo "To use this container with the repository code, run:"
echo ""
echo "      export NEURO_CONTAINER_PATH=\"$FINAL_PATH\""
echo "========================================================"