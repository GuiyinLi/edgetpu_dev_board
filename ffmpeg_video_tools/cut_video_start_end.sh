if [ "$1" == "-h" ]; then
  echo "Usage: `basename $0` {INPUT_VIDEO} {START} {END} {OUTPUT_VIDEO_DIR} {VIDEO_NAME}"
  exit 0
fi

INPUT_VIDEO=$1

START=$2

END=$3

OUTPUT_VIDEO_DIR=$4

VIDEO_NAME=$5

OUTPUT_VIDEO=${OUTPUT_VIDEO_DIR}/cut_${START}_${END}_${VIDEO_NAME}.mp4

ffmpeg -ss ${START} -t ${END} -i ${INPUT_VIDEO} ${OUTPUT_VIDEO}
