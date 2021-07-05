# USE THE WEBCAM
# run inference on a video with DEFAULT TPU MODELS
# WRITE A VIDEO BACK

# metadata on the video to save files with this info
LOCATION=cupertino_roomba
DAY=`date +"%m-%d-%y"`
echo ${DAY}

TIMESTAMP=`date +%H:%M:%S`
echo ${TIMESTAMP}


# where video is located, change per your device
BASE_VIDEO_DIR=/home/mendel/videos

# where DNN models are
BASE_MODEL_DIR=${TPU_CODE_DIR}/DNN_models/google_pretrain/

BASE_LABEL_DIR=${TPU_CODE_DIR}/labels/google_pretrain/

# mobilenet for the edge TPU
MODEL=${BASE_MODEL_DIR}/mobilenet_ssd_v2_coco_quant_postprocess_edgetpu.tflite
CPU_TPU=TPU

# coco label file
LABELS=${BASE_LABEL_DIR}/coco_labels.txt

# anything below this will be filtered out
CONFIDENCE=0.40

# save only top MAX scoring objects
MAX_OBJECTS=7

# prefix to save video
MODEL_NAME=default_MN2

# create a subdirectory with output videos
OUTPUT_VIDEO_DIR=${BASE_VIDEO_DIR}/inference_written_videos/webcam_${MODEL_NAME}/

VIDEO_NUM=location_${LOCATION}_day_${DAY}_timestamp_${TIMESTAMP}

echo ' '
echo 'VIDEO NUM'
echo ${VIDEO_NUM}
echo ' '

OUTPUT_CSV_DIR=${OUTPUT_VIDEO_DIR}
#rm -rf ${OUTPUT_CSV_DIR}
#mkdir -p ${OUTPUT_CSV_DIR}

# do we want video with annotations or raw video to be saved?
# if true, we draw the bounding boxes on top
WRITE_OUT_VIDEO_WITH_PREDICTIONS=True

# do we want to cap the training video to a fixed duration?
# write None if we don't
MAX_VIDEO_DURATION_MINUTES_STR=2

# create a CSV and video too
python3 csv_run_inference.py --model ${MODEL} --labels ${LABELS} --confidence ${CONFIDENCE} --maxobjects ${MAX_OBJECTS} --model_name ${MODEL_NAME} --csv_annotations ${OUTPUT_CSV_DIR} --output-video-dir ${OUTPUT_VIDEO_DIR} --ct $CPU_TPU --out-video-create-mode 'True' --use_webcam 'True' --video_num ${VIDEO_NUM} --write_frame_with_predictions_str ${WRITE_OUT_VIDEO_WITH_PREDICTIONS} --max_video_duration_minutes_str ${MAX_VIDEO_DURATION_MINUTES_STR}
