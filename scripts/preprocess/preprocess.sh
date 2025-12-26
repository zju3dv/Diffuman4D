# require miniconda3
source "$(conda info --base)/etc/profile.d/conda.sh"

DATADIR=""
ACTIONS=()
ALL_ACTIONS=("remove_background" "carve_vhull" "predict_keypoints" "triangulate_skeleton" "draw_skeleton")

while [[ $# -gt 0 ]]; do
  case $1 in
    --data_dir)
      DATADIR=$2
      shift 2
      ;;
    --actions)
      IFS=',' read -r -a ACTIONS <<< "$2"
      shift 2
      ;;
    *)
      echo ">> Unknown arg: $1"; exit 1;;
  esac
done

# run all actions if not specified
if [ ${#ACTIONS[@]} -eq 0 ]; then
  ACTIONS=("${ALL_ACTIONS[@]}")
fi

echo ">> Data directory: $DATADIR"
echo ">> Actions: ${ACTIONS[@]}"

for act in "${ACTIONS[@]}"; do
  case "$act" in
    remove_background)
      conda activate diffuman4d
      python scripts/preprocess/remove_background.py \
        --images_dir "$DATADIR/images" \
        --out_fmasks_dir "$DATADIR/fmasks" \
        --model_name ZhengPeng7/BiRefNet \
        --batch_size 8 # decrease it if OOM
      ;;
    carve_vhull)
      conda activate diffuman4d
      python scripts/preprocess/carve_visual_hull.py \
        --fmasks_dir "$DATADIR/fmasks" \
        --cameras_path "$DATADIR/transforms.json" \
        --out_vhull_dir "$DATADIR/surfs"
      cp "$DATADIR/surfs/000000.ply" "$DATADIR/sparse_pcd.ply"
      ;;
    predict_keypoints)
      # it is recommend to use a seperate conda environment to run sapiens-lite
      # because sapiens-lite requires pytorch<=2.4.1, https://github.com/open-mmlab/mmdetection/issues/12008
      conda activate sapiens_lite
      python scripts/preprocess/predict_keypoints.py \
        --images_dir "$DATADIR/images" \
        --fmasks_dir "$DATADIR/fmasks" \
        --out_kp2d_dir "$DATADIR/poses_sapiens"
      ;;
    triangulate_skeleton)
      conda activate diffuman4d
      python scripts/preprocess/triangulate_skeleton.py \
        --camera_path "$DATADIR/transforms.json" \
        --kp2d_dir "$DATADIR/poses_sapiens" \
        --out_kp3d_dir "$DATADIR/poses_3d" \
        --out_pcd_dir "$DATADIR/poses_pcd" \
        --out_kp2d_proj_dir "$DATADIR/poses_2d"
      ;;
    draw_skeleton)
      conda activate diffuman4d
      python scripts/preprocess/draw_skeleton.py \
        --kp2d_dir "$DATADIR/poses_2d" \
        --out_kpmap_dir "$DATADIR/skeletons"
      ;;
    *)
      echo "Invalid action: $act" >&2
      exit 1
      ;;
  esac
done
