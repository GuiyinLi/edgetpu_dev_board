# wraps a basic script to stream from the edge TPU coral camera to a server
# works quite well!

# note TPU_CODE_DIR is a bash variable pointing to root of this directory

# pretrained models
BASE_MODEL_DIR=${TPU_CODE_DIR}/DNN_models/google_pretrain/

FACENET_MODEL=${BASE_MODEL_DIR}/mobilenet_ssd_v2_face_quant_postprocess_edgetpu.tflite

COCO_SSD_MODEL=${BASE_MODEL_DIR}/mobilenet_ssd_v2_coco_quant_postprocess_edgetpu.tflite

# pretrained labels
BASE_LABEL_DIR=${TPU_CODE_DIR}/labels/google_pretrain/
LABEL_FILE=${BASE_LABEL_DIR}/coco_labels.txt

# for raw facenet
# edgetpu_detect_server --model ${FACENET_MODEL}

# for raw facenet
THRESHOLD=0.5

edgetpu_detect_server --model ${COCO_SSD_MODEL} --labels ${LABEL_FILE} --threshold ${THRESHOLD}
