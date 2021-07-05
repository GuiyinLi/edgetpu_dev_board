# take the ckpt from the latest training and compile for edge TPU

# From the tensorflow/models/research/ directory
TF_MODELS_DIR=/HD1Data/deepcutV2/tensorflow-models/research

# name of final model
NAME=latest_model_joint_waymo_construction_MN2_quantized

BASE_DATA_DIR=/HD1Data/HarvestNet/final_paper_results

# where is the pb file located
MODEL_DIR=${BASE_DATA_DIR}/training_results/train_ssd_mobilenet_v2_coco_quantized_waymo_construction_object_detect_fine_tune/exported_graphs

# checkpoint of mobilenet v2
CHECKPOINT_PATH=$MODEL_DIR/model.ckpt

# where to output converted tflite file
OUTPUT_DIR=$BASE_DATA_DIR/compile_for_edge_tpu/${NAME}
# clear the output dir 
rm -rf ${OUTPUT_DIR}
mkdir -p ${OUTPUT_DIR}

# the pipeline used to train the model in first place
PIPELINE_CONFIG_PATH=$MODEL_DIR/pipeline.config

# got this from tf website
python3 $TF_MODELS_DIR/object_detection/export_tflite_ssd_graph.py \
    --pipeline_config_path=${PIPELINE_CONFIG_PATH} \
    --trained_checkpoint_prefix=${CHECKPOINT_PATH} \
    --output_directory=${OUTPUT_DIR} \
    --add_postprocessing_op=true

## Convert to TFLite
####################
## also from a website, may have to poke around for a final version of this
## but this worked well for paper submission
tflite_convert \
    --output_file=$OUTPUT_DIR/${NAME}.tflite \
    --graph_def_file=$OUTPUT_DIR/tflite_graph.pb \
    --input_arrays='normalized_input_image_tensor' \
    --output_arrays='TFLite_Detection_PostProcess','TFLite_Detection_PostProcess:1','TFLite_Detection_PostProcess:2','TFLite_Detection_PostProcess:3' \
    --input_shapes=1,300,300,3 \
    --allow_custom_ops \
    --inference_type=QUANTIZED_UINT8 \
    --mean_values=128 \
    --std_dev_values=128 \
    --change_concat_input_ranges=false \

# in OUTPUT_DIR: we should see: 
# - tflite_graph.pb
# - recompile_final_paper_joint_waymo_construction_MN2_quantized.tflite
# - ${NAME}.tflite: this works on an embedded CPU

## compile this for the edge TPU
####################
edgetpu_compiler $OUTPUT_DIR/${NAME}.tflite --out_dir $OUTPUT_DIR

## - ${NAME}_edgetpu.tflite: this works on the TPU
