# exact same as custom_retrained_model_detect_demo.sh

# BUT, we use a re-trained model from 3 months after the paper submission and an updated edgetpu_compiler version 2.0.267685300


# note TPU_CODE_DIR is a bash variable pointing to root of this directory
# for dev board
CODE_DIR=/usr/lib/python3/dist-packages/edgetpu/demo/

# for RPI4 + USB accelerator
# CODE_DIR=/usr/share/edgetpu/examples


# what we retrained to detect waymo cars and construction vehicles
MODEL_NAME=latest_model_joint_waymo_construction_MN2_quantized

# since the label file uses this
OLD_MODEL_NAME=final_paper_joint_waymo_construction_MN2_quantized

# custom pretrained models
BASE_MODEL_DIR=${TPU_CODE_DIR}/DNN_models/harvestnet_retrained/rerun_to_check_results
TPU_MODEL=${BASE_MODEL_DIR}/${MODEL_NAME}_edgetpu.tflite
CPU_MODEL=${BASE_MODEL_DIR}/${MODEL_NAME}.tflite

# stock images
BASE_IMAGE_DIR=${TPU_CODE_DIR}/raw_images/raw_images_construction_waymo/

# pretrained labels
BASE_LABEL_DIR=${TPU_CODE_DIR}/labels/harvestnet_retrained/
LABEL_FILE=${BASE_LABEL_DIR}/${OLD_MODEL_NAME}_labels.txt

# where to put output images
BASE_OUT_IMAGE_DIR=${TPU_CODE_DIR}/output_images/harvestnet_retrained
rm -rf ${BASE_OUT_IMAGE_DIR}
mkdir -p ${BASE_OUT_IMAGE_DIR}

# loop over a couple images

for pic_name in `ls ${BASE_IMAGE_DIR}`
do

    python3 $CODE_DIR/object_detection.py \
    --model ${TPU_MODEL} \
    --input ${BASE_IMAGE_DIR}/${pic_name} \
    --label ${LABEL_FILE} \
    --output ${BASE_OUT_IMAGE_DIR}/detect_${pic_name}
done
