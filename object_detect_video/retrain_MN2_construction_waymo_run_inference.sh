# run inference on a video with DEFAULT TPU MODELS

# WRITE A VIDEO BACK

# where video is located, change per your device
VIDEO_DIR=/home/mendel/videos

BASE_MODEL_DIR=${TPU_CODE_DIR}/DNN_models/harvestnet_retrained/

BASE_LABEL_DIR=${TPU_CODE_DIR}/labels/harvestnet_retrained/

NAME=final_paper_joint_waymo_construction_MN2_quantized

# mobilenet for the edge TPU, but retrained!
MODEL=${BASE_MODEL_DIR}/${NAME}_edgetpu.tflite
CPU_TPU=TPU

# mobilenet for the ARM CPU, but retrained!
#MODEL=${BASE_MODEL_DIR}/${NAME}.tflite
#CPU_TPU=CPU

# new label file
LABELS=${BASE_LABEL_DIR}/${NAME}_labels.txt

# anything below this will be filtered out
CONFIDENCE=0.40

# save only top MAX scoring objects
MAX_OBJECTS=7

# prefix to save video
MODEL_NAME=retrained_MN2

for video_num in extra_cut_1686;
#for video_num in new_construction_2811 new_construction_2814 new_construction_2815;
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

