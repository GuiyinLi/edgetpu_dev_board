# note TPU_CODE_DIR is a bash variable pointing to root of this directory
# for dev board
CODE_DIR=/usr/lib/python3/dist-packages/edgetpu/demo/

# for RPI4 + USB accelerator
#CODE_DIR=/usr/share/edgetpu/examples

# pretrained models
BASE_MODEL_DIR=${TPU_CODE_DIR}/DNN_models/google_pretrain/

# stock images
BASE_IMAGE_DIR=${TPU_CODE_DIR}/raw_images/

# pretrained labels
BASE_LABEL_DIR=${TPU_CODE_DIR}/labels/google_pretrain/

BASE_OUT_IMAGE_DIR=${TPU_CODE_DIR}/output_images/

python3 $CODE_DIR/object_detection.py \
--model ${BASE_MODEL_DIR}/mobilenet_ssd_v2_face_quant_postprocess_edgetpu.tflite \
--input ${BASE_IMAGE_DIR}/face.jpg \
--output ${BASE_OUT_IMAGE_DIR}/face_detections.jpg

python3 $CODE_DIR/object_detection.py \
--model ${BASE_MODEL_DIR}/mobilenet_ssd_v2_coco_quant_postprocess_edgetpu.tflite \
--input ${BASE_IMAGE_DIR}/parrot.jpg \
--label ${BASE_LABEL_DIR}/coco_labels.txt \
--output ${BASE_OUT_IMAGE_DIR}/parrot_detections.jpg


