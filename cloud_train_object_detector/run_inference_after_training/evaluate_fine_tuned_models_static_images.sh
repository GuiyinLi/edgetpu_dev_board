# where is the validation data located?
# INPUT_IMAGES_DIR=/HD1Data/HarvestNet/dataset/${NAME}/${label}/${SET}

DATA_NAME=waymo_construction
# testset
SET=val

# where data and label files are 
BASE_DATA_DIR=/HD1Data/HarvestNet/final_paper_results/

# where raw images for training and validation go
BASE_IMAGES_DIR=${BASE_DATA_DIR}/raw_image_training_val_data/${DATA_NAME}

# where training results are
RESULTS_DIR=${BASE_DATA_DIR}/training_results/

# what DNN model to use
# example if we just want to run 1
DNN_MODELS=('ssd_mobilenet_v2_coco_quantized')

# case where we want to run both
#DNN_MODELS=('ssd_mobilenet_v2_coco_quantized' 'faster_rcnn_resnet101')

# evaluation code
for MODEL_NAME in ${DNN_MODELS[@]}; do

    echo "running model: ${MODEL_NAME}\n"

    # where the re-trained model checkpoints go
    MODEL_DIR=${RESULTS_DIR}/train_${MODEL_NAME}_${DATA_NAME}_object_detect_fine_tune

    # where the frozen model graph is
    FROZEN_GRAPH_PATH=${MODEL_DIR}/exported_graphs/frozen_inference_graph.pb
    echo ${FROZEN_GRAPH_PATH}

    # where labels are
    LABELS_PATH=/HD1Data/HarvestNet/final_paper_results/labels/${DATA_NAME}_new/label_map.pbtxt

    echo ${LABELS_PATH}

    # where results go
    OUTPUT_DIR=$BASE_DATA_DIR/evaluation_results/evaluate_${DATA_NAME}_${SET}_${MODEL_NAME}
    rm -rf ${OUTPUT_DIR}
    mkdir -p ${OUTPUT_DIR}
    echo ${OUTPUT_DIR}

    # make this -1 if we wanna run all images
    MAX_IMAGES=-1

    # which dataset images to loop over
    LABELS=('construction' 'waymo')

    # which GPU to use
    GPU_NUM=3

    # evaluation code
    for label in ${LABELS[@]}; do
        # input images
        INPUT_IMAGES_DIR=${BASE_IMAGES_DIR}/${label}/${SET}
        echo ${INPUT_IMAGES_DIR}

        # place images with annotations overlaid here
        OUTPUT_IMAGES_DIR=${OUTPUT_DIR}/${label}
        rm -rf ${OUTPUT_IMAGES_DIR}
        mkdir -p ${OUTPUT_IMAGES_DIR}
        echo ${OUTPUT_IMAGES_DIR}
        
        # loop thru images, draw annotations and write csv of results to calculate mAP later
        export CUDA_VISIBLE_DEVICES=${GPU_NUM} && python3 faster_pretrained_object_detector.py --input-images-dir $INPUT_IMAGES_DIR --output-images-dir $OUTPUT_IMAGES_DIR --model-name $MODEL_NAME --frozen-graph-path $FROZEN_GRAPH_PATH --labels-path $LABELS_PATH --max-images ${MAX_IMAGES}
    done

done
