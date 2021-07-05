MODEL_FILE=${TPU_CODE_DIR}/DNN_models/google_pretrain/deeplabv3_mnv2_dm05_pascal_quant_edgetpu.tflite

INPUT_PIC_DIR=${TPU_CODE_DIR}/raw_images/raw_images_construction_waymo

for INPUT_PIC in `ls ${INPUT_PIC_DIR}`
do
    echo ${INPUT_PIC_FNAME}

    INPUT_PIC=${INPUT_PIC_DIR}/${INPUT_PIC}
    echo ${INPUT_PIC}
    OUTPUT_PIC=${INPUT_PIC_DIR}/segmented_${INPUT_PIC}
    echo ${OUTPUT_PIC}

	python3 semantic_segmentation_example_google.py --model ${MODEL_FILE} --input ${INPUT_PIC} --output ${OUTPUT_PIC}
done

