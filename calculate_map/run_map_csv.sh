# loop thru test images to calculate a csv of predictions

# in another script, compared with ground truth to ge the mAP metric
BASE_MODEL_DIR=${TPU_CODE_DIR}/DNN_models/harvestnet_retrained/

BASE_LABEL_DIR=${TPU_CODE_DIR}/labels/harvestnet_retrained/

NAME=final_paper_joint_waymo_construction_MN2_quantized

# TPU MODEL
#MODEL=${BASE_MODEL_DIR}/${NAME}_edgetpu.tflite
#PLATFORM=TPU

# comment this or the above
# CPU MODEL (there is a slight difference)
MODEL=${BASE_MODEL_DIR}/${NAME}.tflite
PLATFORM=CPU

LABELS=${BASE_LABEL_DIR}/${NAME}_labels.txt

CONFIDENCE=0.50
MAX_OBJECTS=500

MODEL_NAME=final_paper_retrain_${PLATFORM}_MN2

PRINT_MODE=True

# can be slow, do all the images
MAX_IMAGES=-1

# do first 30 images
#MAX_IMAGES=10

# where images are
BASE_IMAGES_DIR=/mnt/validation_images_final_paper_waymo_construction/waymo_construction/joint_data/

OUTPUT_IMAGES_DIR=/mnt/validation_images_final_paper_waymo_construction/${PLATFORM}_annotated_images/
rm -rf ${OUTPUT_IMAGES_DIR}
mkdir -p ${OUTPUT_IMAGES_DIR}

python3 get_csv_for_map_calc.py --base-images-dir ${BASE_IMAGES_DIR} --output-images-dir ${OUTPUT_IMAGES_DIR} --model ${MODEL} --labels ${LABELS} --confidence ${CONFIDENCE} --maxobjects ${MAX_OBJECTS} --model_name ${MODEL_NAME} --print-mode ${PRINT_MODE} --max_images $MAX_IMAGES

