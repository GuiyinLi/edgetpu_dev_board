# note TPU_CODE_DIR is a bash variable pointing to root of this directory

# for dev board
CODE_DIR=/usr/lib/python3/dist-packages/edgetpu/demo/

# for RPI4 + USB accelerator
# CODE_DIR=/usr/share/edgetpu/examples

# pretrained models
BASE_MODEL_DIR=${TPU_CODE_DIR}/DNN_models/google_pretrain/

# stock images
BASE_IMAGE_DIR=${TPU_CODE_DIR}/raw_images/

# pretrained labels
BASE_LABEL_DIR=${TPU_CODE_DIR}/labels/google_pretrain/

python3 $CODE_DIR/classify_image.py \
--model ${BASE_MODEL_DIR}/mobilenet_v2_1.0_224_inat_bird_quant_edgetpu.tflite \
--label ${BASE_LABEL_DIR}/inat_bird_labels.txt \
--image ${BASE_IMAGE_DIR}/parrot.jpg

