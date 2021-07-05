# mobilenet V2

### This code fine-tunes a tensorflow object detector with new data and output classes
### meant to be run on a cloud GPU server

# where the model checkpoint and training data is, configure for your machine
BASE_DATA_DIR=/HD1Data/HarvestNet/final_paper_results/

# base dir where results go: change per your machine
RESULTS_DIR=${BASE_DATA_DIR}/training_results/

# DNN type
MODEL_NAME=ssd_mobilenet_v2_coco_quantized

# suffix to add to files to name the model
# our case: joint waymo and construction data
DATA_NAME=waymo_construction

# configuration file to train the object detector
# make sure to change this to have paths relevant to your dataset and machine
PIPELINE_CONFIG_PATH=configs/${MODEL_NAME}_${DATA_NAME}.config

# where the re-trained model checkpoints go
MODEL_DIR=${RESULTS_DIR}/train_${MODEL_NAME}_${DATA_NAME}_object_detect_fine_tune
rm -rf ${MODEL_DIR}
mkdir -p ${MODEL_DIR}

# how long to run, nominally 50k
NUM_TRAIN_STEPS=30000
SAMPLE_1_OF_N_EVAL_EXAMPLES=1

# which GPU to train on: our machine has 0-3
GPU_NUM=3

# standard code from tensorflow examples to re-train the object detector
export CUDA_VISIBLE_DEVICES=${GPU_NUM} &&
python3 $TF_MODELS_DIR/object_detection/model_main.py \
    --pipeline_config_path=${PIPELINE_CONFIG_PATH} \
    --model_dir=${MODEL_DIR} \
    --num_train_steps=${NUM_TRAIN_STEPS} \
    --sample_1_of_n_eval_examples=$SAMPLE_1_OF_N_EVAL_EXAMPLES \
    --alsologtostderr
