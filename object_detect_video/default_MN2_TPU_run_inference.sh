# run inference on a video with DEFAULT TPU MODELS

# WRITE A VIDEO BACK

# where video is located, change per your device
# for the dev board
# VIDEO_DIR=/home/mendel/videos

# for using small videos stored in this repo [fully self-contained]
VIDEO_DIR=/home/mendel/videos
#VIDEO_DIR=${TPU_CODE_DIR}/sample_video/

BASE_MODEL_DIR=${TPU_CODE_DIR}/DNN_models/google_pretrain/

BASE_LABEL_DIR=${TPU_CODE_DIR}/labels/google_pretrain/

# mobilenet for the edge TPU
MODEL=${BASE_MODEL_DIR}/mobilenet_ssd_v2_coco_quant_postprocess_edgetpu.tflite
CPU_TPU=TPU

# coco label file
LABELS=${BASE_LABEL_DIR}/coco_labels.txt

# anything below this will be filtered out
CONFIDENCE=0.60

# save only top MAX scoring objects
MAX_OBJECTS=10

# prefix to save video
MODEL_NAME=default_MN2

for video_num in extra_cut_1686;
do

    BASE_VIDEO_DIR=${VIDEO_DIR}

    # create a subdirectory with output videos
    OUTPUT_VIDEO_DIR=${BASE_VIDEO_DIR}/inference_written_videos/${MODEL_NAME}_${video_num}/
    rm -rf ${OUTPUT_VIDEO_DIR}
    mkdir -p ${OUTPUT_VIDEO_DIR}

    OUTPUT_CSV_DIR=${OUTPUT_VIDEO_DIR}

    # MUCH FASTER, create a CSV only
    #python3 csv_run_inference.py --video_num $video_num --base-video-dir ${BASE_VIDEO_DIR} --model ${MODEL} --labels ${LABELS} --confidence ${CONFIDENCE} --maxobjects ${MAX_OBJECTS} --model_name ${MODEL_NAME} --csv_annotations ${OUTPUT_CSV_DIR} --output-video-dir ${OUTPUT_VIDEO_DIR} --ct $CPU_TPU

    # MUCH SLOWER, create a CSV and video too
    python3 csv_run_inference.py --video_num $video_num --base-video-dir ${BASE_VIDEO_DIR} --model ${MODEL} --labels ${LABELS} --confidence ${CONFIDENCE} --maxobjects ${MAX_OBJECTS} --model_name ${MODEL_NAME} --csv_annotations ${OUTPUT_CSV_DIR} --output-video-dir ${OUTPUT_VIDEO_DIR} --ct $CPU_TPU --out-video-create-mode 'True'

done

