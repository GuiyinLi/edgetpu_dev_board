# exact same as run_few_images_save_annotations.sh

# BUT, we use a re-trained model from 3 months after the paper submission and an updated edgetpu_compiler version 2.0.267685300

# STILL the models work, showing robustness of the training procedure

# loop thru test images to calculate a csv of predictions

# in another script, compared with ground truth to ge the mAP metric
BASE_MODEL_DIR=${TPU_CODE_DIR}/DNN_models/harvestnet_retrained/rerun_to_check_results

BASE_LABEL_DIR=${TPU_CODE_DIR}/labels/harvestnet_retrained/

# new model re-trained much after paper submit
NAME=latest_model_joint_waymo_construction_MN2_quantized

# since the label file uses this
OLD_MODEL_NAME=final_paper_joint_waymo_construction_MN2_quantized

# TPU MODEL
MODEL=${BASE_MODEL_DIR}/${NAME}_edgetpu.tflite
PLATFORM=TPU

LABELS=${BASE_LABEL_DIR}/${OLD_MODEL_NAME}_labels.txt

CONFIDENCE=0.50
MAX_OBJECTS=500

MODEL_NAME=final_paper_retrain_TPU_MN2

PRINT_MODE=True

# can be slow, do all the images
MAX_IMAGES=-1

# where images are
BASE_IMAGES_DIR=${TPU_CODE_DIR}/raw_images/raw_images_construction_waymo/

OUTPUT_IMAGES_DIR=${TPU_CODE_DIR}/output_images/${PLATFORM}_annotated_images/
rm -rf ${OUTPUT_IMAGES_DIR}
mkdir -p ${OUTPUT_IMAGES_DIR}


# write a csv of predictions
python3 get_csv_for_map_calc.py --base-images-dir ${BASE_IMAGES_DIR} --output-images-dir ${OUTPUT_IMAGES_DIR} --model ${MODEL} --labels ${LABELS} --confidence ${CONFIDENCE} --maxobjects ${MAX_OBJECTS} --model_name ${MODEL_NAME} --print-mode ${PRINT_MODE} --max_images $MAX_IMAGES

