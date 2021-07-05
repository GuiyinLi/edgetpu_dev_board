## Purpose

This repo provides pre-trained object detector models to recognize construction sites and self-driving cars on the road, part of the HarvestNet project at Stanford.
These models are for Tensorflow Lite and for the Google Edge TPU Dev Board.

Further details on the project can be found here: https://sites.google.com/view/harvestnet/home

We modified a lot of demo scripts and pre-trained models from Coral, present here: https://coral.ai/docs/

Please direct any questions to : csandeep [at] stanford.edu




## Preliminaries
in your bashrc add a line pointing to where this repo is located, e.g:
```sh
export TPU_CODE_DIR=/home/mendel/edgetpu_dev_board_release/
```

- this code uses the edge TPU dev board, with software updates BEFORE sept 2019
	- python3
	- edge TPU API version 2.11.1
	- board was flashed using Mendel Development Tool (MDT) version 1.3
	- device name and IP: indigo-snail		(192.168.100.2)
	- check in python3 via: 

```sh
import edgetpu
edgetpu.__version__ 
>>> '2.11.1'
``` 

## Code Structure
The following are key subdirectories
##### DNN models/
- pretrained models from Coral and our quantized re-trained models to recognize construction sites and waymo cars

##### calculate_map/ 
- code to calculate the mean average precision (mAP) of our retrained models and run the models on validation images

##### labels/
- label files for object detector output classes

##### object_detect_video/
- code to run the TPU models on a video to do live inferencing, saves output video and a CSV with inference results 

##### raw_images/
- a few images to try the models on

##### sample_video/
- a short 5-10 sec video we captured to test the models on

##### verify_setup/
- first try this code to see the TPU inferencing works well

##### split_video_into_images/
- extract a few images from a video 


## Board Setup

### Connect to Dev Board
- type mdt shell on your home computer (eg Mac)
- name of our device: indigo-snail

### On the Dev Board

- source ~/.bashrc
	- you should see color in your prompt
	
- type workon cv to enter the opencv python virtual env, or install opencv yourself
- run "edgetpu_demo --stream" from anywhere on device
	- go to: 192.168.100.2:4664 in your web browser
	- you should see the video stream with live inferencing
- all our code is in the edgetpu_dev_board_release repo (this repo)

- check the basic demo scripts under the verify_setup/ subdir
	- run_classify_demo.sh
	- object_detect_demo.sh
	- these are essentially copied from the Coral website

- if you get this error, means two processes are trying to access the TPU 
```sh
RuntimeError: Error in device opening (/dev/apex_0)!
``` 
- at this point the TPU is working, now lets look at our code
	- where are videos? 
		- **/home/mendel/videos or edgetpu_dev_board_release/sample_videos/**
	- where are our re-trained models 
		- **edgetpu_dev_board_release/DNN_models/harvestnet_retrained**
- where are our label files: 
		- **edgetpu_dev_board_release/labels/harvestnet_retrained**

- how can I run inference on our saved videos with the default MN2? 
	- object_detect_video/default_MN2_TPU_run_inference.sh

- with our re-trained models? 
	- object_detect_video/retrain_MN2_construction_waymo_run_inference.sh



### how to mount the external storage:
``` sh
$ sudo /sbin/fdisk -l
assuming your SD card is /dev/sdc1
$ sudo mount /dev/mmcblk1p1 /mnt
```

### how to push/pull data from your mac
- mdt push/pull
- mdt push source dest

### how to off the device safely
- sudo shutdown now

### How to Connect to the Dev Board via WiFi only
- on your host computer: 

``` sh
$ mdt devices
indigo-snail		(192.168.0.22)
```

- then: ssh mendel@indigo-snail OR mendel@<IP-ADDR>

- further links here: https://medium.com/@aallan/hands-on-with-the-coral-dev-board-adbcc317b6af


# Acknowledgements
We heavily modified publicly available code from here in the object_detection/ subdir:

https://www.bouvet.no/bouvet-deler/hands-on-with-the-google-coral-usb-accelerator





