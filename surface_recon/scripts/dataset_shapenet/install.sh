source dataset_shapenet/config.sh

# Function for processing a single model
reorganize_choy2016() {
  modelname=$(basename -- $5)
  output_path="$4/$modelname"
  build_path=$3
  choy_vox_path=$2
  choy_img_path=$1

  points_non_file="$build_path/4_points/$modelname.npz"
  points_non_out_file="$output_path/points_watertight.npz"

  pointcloud_non_file="$build_path/5_points/$modelname.npz"
  pointcloud_non_out_file="$output_path/points_non_watertight.npz"


  img_dir="$choy_img_path/$modelname/rendering"
  img_out_dir="$output_path/img_choy2016"

  metadata_file="$choy_img_path/$modelname/rendering/rendering_metadata.txt"
  camera_out_file="$output_path/img_choy2016/cameras.npz"

  mkdir -p $output_path $img_out_dir

  mv $points_non_file $points_non_out_file
  mv $pointcloud_non_file $pointcloud_non_out_file
}

export -f reorganize_choy2016

# Make output directories
mkdir -p $OUTPUT_PATH

CHOY2016_PATH="../data/external/Choy2016"

# Run build
for c in ${CLASSES[@]}; do
  BUILD_PATH_C=$BUILD_PATH/$c
  OUTPUT_PATH_C=$OUTPUT_PATH/$c
  CHOY2016_IMG_PATH_C="$CHOY2016_PATH/ShapeNetRendering/$c"
  CHOY2016_VOX_PATH_C="$CHOY2016_PATH/ShapeNetVox32/$c"
  mkdir -p $OUTPUT_PATH_C

  ls $CHOY2016_VOX_PATH_C | parallel -P $NPROC --timeout $TIMEOUT \
    reorganize_choy2016 $CHOY2016_IMG_PATH_C $CHOY2016_VOX_PATH_C \
      $BUILD_PATH_C $OUTPUT_PATH_C {}

  echo "Creating split"
   python create_split.py $OUTPUT_PATH_C --r_val 0.1 --r_test 0.2
done
