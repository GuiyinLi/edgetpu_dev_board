# code to export a frozen graph for a fine-tuned model
# same procedure as mobilenet v2, just change the MODEL_NAME

BASE_DATA_DIR=/HD1Data/HarvestNet/final_paper_results/

# DNN type
MODEL_NAME=faster_rcnn_resnet101

# suffix to add to files to name the model
# our case: joint waymo and construction data
DATA_NAME=waymo_construction

# configuration file to train the object detector
PIPELINE_CONFIG_PATH=../configs/${MODEL_NAME}_${DATA_NAME}.config

# base dir where results go: change per your machine
RESULTS_DIR=${BASE_DATA_DIR}/training_results/

# where the re-trained model checkpoints go
MODEL_DIR=${RESULTS_DIR}/train_${MODEL_NAME}_${DATA_NAME}_object_detect_fine_tune

# this has to be manually set, typically see the ckpt with the highest VALIDATION mAP
# can see this in tensorboard, corresponds to a ckpt in MODEL_DIR
# for example, we can see in MODEL_DIR: model.ckpt-2000.index
CHECKPOINT_NUMBER=2000

python3 $TF_MODELS_DIR/object_detection/export_inference_graph.py \
    --input_type image_tensor \
    --pipeline_config_path $PIPELINE_CONFIG_PATH \
    --trained_checkpoint_prefix $MODEL_DIR/model.ckpt-${CHECKPOINT_NUMBER} \
    --output_directory $MODEL_DIR/exported_graphs
