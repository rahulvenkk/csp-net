ROOT=..

DATA_ROOT=..

export MESHFUSION_PATH=$ROOT/external/mesh-fusion
export HDF5_USE_FILE_LOCKING=FALSE # Workaround for NFS mounts

INPUT_PATH=$DATA_ROOT/data/external/ShapeNetCore.v1
BUILD_PATH=$DATA_ROOT/data/ShapeNet_full.build
OUTPUT_PATH=$DATA_ROOT/data/ShapeNet_full

NPROC=4
TIMEOUT=180
N_MODELS=67780

declare -a CLASSES=(
03001627 # chair
# 02958343 # car
# 04379243 # table
#04256520 # sofa
#02691156 # planes
#03636649 # lamp
#04401088
#04530566
#03691459
#02933112
#03211117
#02828884
#04090263
)


lsfilter() {
 folder=$1
 other_folder=$2
 ext=$3
 COUNTER=0
 for f in $folder/*; do
   filename=$(basename $f)
   if [ ! -f $other_folder/$filename$ext ] && [ ! -d $other_folder/$filename$ext ]; then
    echo $filename
   fi
   COUNTER=$[$COUNTER +1]
   if (($COUNTER >= $N_MODELS)); then
    break
   fi
 done
}
