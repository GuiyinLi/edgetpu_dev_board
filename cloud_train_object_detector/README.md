## Purpose
Code to fine-tune an object detector CNN in tensorflow using a new set of labels.
This code fine-tunes faster RCNN and a quantized MobileNet v2, which will eventually be compiled for the Google Edge TPU.

## Prerequisites

***
python3

tensorflow-gpu version 1.12.0

```python
>>> import tensorflow
>>> tensorflow.__version__
'1.12.0'
```

we used a virtual env called v2-env

ensure you have a working installation of the tensorflow object detection API

1. https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md
2. remember the protobuffers part and coco api tools installation listed above
3. we used upto this commit from google:

```sh
        commit 8c232a478d5d6d5c6eeacd05056568662069a1fb
        Author: Jonathan Mitchell <Jmitchell1991@gmail.com>
        Date:   Wed Apr 17 10:17:32 2019 -0700
            Adds original mobilenetv2 keras implementation to Object Detection (#6592)
```

have the following variables in your bashrc, example: 

```sh
export TF_MODELS_DIR='/HD1Data/deepcutV2/tensorflow-models/research'
export PYTHONPATH=$PYTHONPATH:${TF_MODELS_DIR}:${TF_MODELS_DIR}/slim 
export HARVESTNET_ROOT_DIR='/Users/csandeep/Documents/work/uhana/work/harvestnet_implement
```

change these as per your local structure, base directory for this repo
***

## Assemble Training Data and Model Checkpoints:

for our server, see: **/HD1Data/HarvestNet/final_paper_results/**

for a new dataset, we assume the following layout:
    - labels/ 
        - a label file of output classes (construction and self-driving cars for our case)
        
    - pretrain_tensorflow_checkpoints/
        - .pb files and .ckpt files for faster-RCNN and mobilenet v2 quantized
        - downloaded from Tensorflow Model Zoo
        - available at : *https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md*
        - model versions are subdirectories, example:
            - faster_rcnn_resnet101_coco_11_06_2017/
            - ssd_mobilenet_v2_quantized_300x300_coco_2019_01_03/
	- train_configs/
		- config files used to fine-tune and train the tensorflow models
		- base configurations were downloaded from Google
		- paths to our training data, label files, and changes to number of output labels need to be made (see our examples)
		
		- training_data_tfrecords/
        - training and validation tensorflow record files
        - have images we annotated using the Google Cloud Labeling Service
        - need to re-generate this for your new application

        in our case:
        - construction_new_train.record
        - construction_new_val.record
        - waymo_new_train.record
        - waymo_new_val.record

## Check GPU and TF Object Detection API are working correctly

0.    python3 check_GPU.py

    - should see a long output ending with: 
	- Default GPU Device: /device:GPU:0

1. check that TFLOW OBJ DET API is working well
    ./test_install.sh
    - should see something like:

    - Ran 16 tests in 0.056s
      OK

## Fine-tune (Train) the object detector 

Run these in a screen, since they may take long

1. update the config file for the model you are using to have the correct paths
	- see README in the config/ folder
	- example for MN2: modified_ssd_mobilenet_v2_coco_QUANTIZED_waymo_construction.config
    - as per readme, change graph_rewriter delay to NUM_STEPS (see below) - 2k or so
    - e.g if we train for 30k steps, we start quantization after 28k steps 

2. in a screen, start the following process change the GPU device number to what you want to use
    - change NUM_STEPS in following script: can be 30k or 50k
    - quantize_waymo_construction_fine_tune.sh
    - this creates new model ckpts for the fine-tuned models and tensorflow logs here:
        - ${RESULTS_DIR}/train_waymo_construction_object_detect_fine_tune/
        - ${RESULTS_DIR} on our machine: **/HD1Data/HarvestNet/final_paper_results/training_results/**
        - model ckpts example:
            - *model.ckpt-2000.index*
        - tensorboard events log (used to visualize training procedure):
            - *events.out.tfevents.1576623809.ubuntu16*

    - if this is running correctly, you will see the validation accuracy printed to the screen after a long time and intermittently, like:

    - exact same process to fine tune faster-rcnn
        ./mrcnn_waymo_construction_fine_tune.sh*

    - our models do OK even after 2k steps, which takes about 20 mins on a GPU
    - for the paper, we trained them for 50k steps

## Visualize Training Using Tensorboard 
0. on the GPU where the model is training, in the ${RESULTS_DIR} where the 'events' file is located, run:
    - tensorboard --logdir . --port 6006

1. from your local machine where you want to view tensorboard in the browser run:
    - ssh -N -f -L localhost:16006:localhost:6006 <user@remote>

2. go to (in this case) http://localhost:16006 on your local machine

    - blue color: notice the validation losses take time to show up
    - orange color: training loss
    - see the images tab to see sample detections from the model

## Export a Frozen Fine-Tuned Model

	- see subdir: export_frozen_model_post_train/
    - for mobilenet v2, run: export_fine_tune_waymo_construction.sh
    - manually adjust CHECKPOINT_NUMBER to the last or best validation accuracy checkpoint
    - the frozen model will be at MODEL_DIR/exported_graphs/
        - frozen_inference_graph.pb (around 20 MB size for MN2, 180 MB for RCNN)

## Test new object detector on specific images

    test the new object detector on a number of static images
    - see the run_inference_after_training/ subdir

    - test a frozen, fine-tuned model, draw annotated images and create csv
        - evaluate_fine_tuned_models_static_images.sh

    - run same images BUT with original MobileNet or other model that is unmodified [no custom labels]
        - default_models_evaluate_static_images.sh


## Quantize mobilenet and compile for the edge TPU
    - subdir: quantize_for_edge_tpu

    - converts a model to tflite by quantizing to INT8
    - then compiles for edge TPU, using edgetpu_compiler binary

    - to compile new models we trained:
        - new_models_quantize_and_compile_TPU.sh
    - to quantize a model from a ckpt we saved from the harvestnet paper:
        - quantize_and_compile_TPU_final_paper_models.sh

    some useful links:
    https://coral.withgoogle.com/docs/edgetpu/compiler/
    https://thenewstack.io/train-and-deploy-tensorflow-models-optimized-for-google-edge-tpu/

    STEP 1: convert frozen pb file to quantized tflite

        - converts to int8

        - creates a final model called final_paper_joint_waymo_construction_MN2_quantized_edgetpu.tflite (for example)

    STEP 2: convert quantized tflite to one that works on EDGETPU using edgetpu_compiler

        - goes to the directory with the .tflite model

        - runs edgetpu_compiler from the command line 

        - this creates a new model with the _edgetpu SUFFIX

    An example of the log that occurs after compilation:

    On-chip memory available for caching model parameters: 7.62MiB
    On-chip memory used for caching model parameters: 5.21MiB
    Off-chip memory used for streaming uncached model parameters: 0.00B

## Acknowledgements

We wrapped a lot of public code from the official tensorflow documentation, available here: https://github.com/tensorflow/models/tree/master/research/object_detection

