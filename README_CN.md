# Edge TPU DEV Board

## 软件环境

- python3.7
- edge TPU API version 2.11.1
- opencv4.5.2
- PIL

## 硬件环境

- Coral加速棒 - TPU
- RaspberryPi 4B - CPU
- Camera(处理录制好的视频可以不用)
- Mac/Linux操作系统
- [设备图](./Board.png)

## this code uses the edge TPU dev board, with software updates BEFORE sept 2019

- python3
- edge TPU API version 2.11.1
- board was flashed using Mendel Development Tool (MDT) version 1.3
- device name and IP: indigo-snail(192.168.100.2)
- check in python3 via:

- [介绍网站](https://zhuanlan.zhihu.com/p/65813704?utm_source=wechat_session)
- [edgetpu API Docs](https://coral.ai/docs/edgetpu/api-intro/#install-the-library)

<div STYLE="page-break-after: always;"></div>

# 软件环境配置

## 测试设备

```sh
Editor:		Visual Studio Code 1.57.1
Dev Board: 	RaspberryPi 4B 4G rev1.2
connect: 	Remote-SSH
```

## 测试环境

```sh
Raspberry Pi 4B 4G
Debian Linux(和Ubuntu差不多)
python 3.7.3
pip3 21.1.2
```

## 版本验证

```sh
python3 -V
pip3 -V
```

## 更新和依赖安装

```sh
sudo apt-get update
sudo apt-get upgrade
sudo apt-get install libhdf5-dev libhdf5-serial-dev
sudo apt-get install libqtgui4 libqtwebkit4 libqt4-test python3-pyqt5
sudo apt-get install libatlas-base-dev
sudo apt-get install libjasper-dev
python3 -m pip install --upgrade pip
python3 -m pip install numpy
```

## 安装opencv

```sh
python3 -m pip install opencv-python
python3 -m pip install opencv-contrib-python
```

## 安装PIL

```sh
python3 -m pip  install Pillow
```

## 安装edegtpu

```sh
wget https://dl.google.com/coral/edgetpu_api/edgetpu_api_2.11.1.tar.gz -O edgetpu_api_2.11.1.tar.gz --trust-server-names
tar xzf edgetpu_api_2.11.1.tar.gz
cd edgetpu_api/
sudo bash ./install.sh
```

## 验证edgetpu版本

```sh
edgetpu.__version__
```

## 修复'_edgetpu_cpp_wrapper'

```sh
find /usr/local/ -name \*edgetpu\*
cd /usr/local/lib/python3.7/dist-packages/edgetpu/swig/
sudo ln -s _edgetpu_cpp_wrapper.cpython-35m-aarch64-linux-gnu.so _edgetpu_cpp_wrapper.cpython-37m-aarch64-linux-gnu.so
```

## 安装验证

```py
import edgetpu,cv2,PIL
edgetpu.__version__,cv2.__version__,PIL.__version__
>>>('2.11.1', '4.5.2', '8.2.0')
```

<div STYLE="page-break-after: always;"></div>

# Code Structure

- 包括mAP计算程序、云服务器训练代码、标注、采样视频和图片、预测程序、预测结果输出文件、安装校验代码、以及一些工具(ffmpeg推流、segmentation图像分割、split视频帧截取等)。
每个文件夹主要是py文件+sh运行脚本

## DNN models/

- 预测模型，包括谷歌官方和开发者自己训练的

## calculate_map/

- 用于计算mAp值，它用于衡量预测效果，范围0-1，值越高越好

## labels/

- 标注信息，用于设置数据集，参与训练

## object_detect_video/

- 核心预测程序，用于实时预测视频流或者预测已经录制好的视频并输出，包含预测程序py和执行程序脚本sh

## raw_images/

- a few images to try the models on

## sample_video/

- a short 5-10 sec video we captured to test the models on,用于object_detect_video中的预测程序

## verify_setup/

- first try this code to see the TPU inferencing works well，执行里面的脚本文件开启自动识别并输出图片，从而验证软硬件环境是否配置完毕

## split_video_into_images/

- extract a few images from a video，截取视频帧存为图片

<div STYLE="page-break-after: always;"></div>

# Running Config

## 修复Win10中脚本sh文件空行BUG

```sh
sudo apt-get  install dos2unix
dos2unix dos2unix retrain_MN2_construction_waymo_run_inference.sh
dos2unix dos2unix *.sh
```

## 将项目路径添加至环境变量

```sh
export TPU_CODE_DIR=/home/pi/Work/edgetpu_dev_board_release
```

## 修改可执行脚本

```txt
位于TPU_CODE_DIR/object_dectect_video/retrain_MN2_construction_waymo_run_inference.sh
```

## csv_run_inference.py参数解释

| Syntax        | Description   | Deaufltvalue | Setvalue     |
| ------------- | ------------- | ------------ | ------------ |
| --base-video-dir | 目标视频路径    |  | sample_video |
| --video_num      | 视频号/视频名称 |  |extra_cut_1686|
| --output-video-dir | 预测视频输出路径 | | output_video |
| --model_name   | 模型名称    |       | |
| --model        | .tflite文件 | TPU模型    | CPU模型 |
| --labels       | 标注信息所在路径          | | labels |
| --maxobjects   | 每帧视频所检测的目标最大数量 | 3      | 7 |
| --confidence   | 标注目标最小阈值 | 0.6   | 0.4 |
| --ct           | CPU/TPU         | TPU  | CPU  |
| --print-mode   | 打印模式         | True | True |
| --out-video-create-mode | 是否保存预测后的视频 | False | True |
| --csv_annotations| annotations输出路径 | | output_video |
| --use_webcam     | Web在线查看   | False | False        |
| --write_frame_with_predictions_str|输出带有边框的视频|True|True|
| --max_video_duration_minutes_str| |None|None|

<div STYLE="page-break-after: always;"></div>

# [Running Code](./runningpart.png)

## Problem

- 目前还不知道如何修改[csv_run_inference.py](./object_detect_video/csv_run_inference.py)使其可以运行CPU模型

## [脚本命令执行](./object_detect_video/retrain_MN2_construction_waymo_run_inference.sh)

```sh
sh retrain_MN2_construction_waymo_run_inference.sh
```

## python3语句执行

```sh
cd ~/Work/edgetpu_dev_board_release/object_detect_video
```

```sh
python3 csv_run_inference.py \
--base-video-dir ~/Work/edgetpu_dev_board_release/sample_video \
--video_num extra_cut_1686 \
--output-video-dir ~/Work/edgetpu_dev_board_release/output_images \
--model ~/Work/edgetpu_dev_board_release/DNN_models/harvestnet_retrained/final_paper_joint_waymo_construction_MN2_quantized.tflite \
--labels ~/Work/edgetpu_dev_board_release/labels/harvestnet_retrained/final_paper_joint_waymo_construction_MN2_quantized_labels.txt \
--ct CPU \
--out-video-create-mode True \
--csv_annotations ~/Work/edgetpu_dev_board_release/output_images
```

```sh
cd ~/Work/edgetpu_dev_board_release/output_video
```

## Result
```txt
args:  {'base_video_dir': '/home/pi/Work/edgetpu_dev_board_release/sample_video', 'video_num': 'extra_cut_1686', 'output_video_dir': '/home/pi/Work/edgetpu_dev_board_release/output_video/', 'model_name': 'retrained_MN2', 'model': '/home/pi/Work/edgetpu_dev_board_release/DNN_models/harvestnet_retrained/final_paper_joint_waymo_construction_MN2_quantized.tflite', 'labels': '/home/pi/Work/edgetpu_dev_board_release/labels/harvestnet_retrained/final_paper_joint_waymo_construction_MN2_quantized_labels.txt', 'maxobjects': 7, 'confidence': 0.4, 'ct': 'CPU', 'print_mode': 'False', 'out_video_create_mode': 'True', 'csv_annotations': '/home/pi/Work/edgetpu_dev_board_release/output_video/', 'use_webcam': 'False', 'write_frame_with_predictions_str': 'True', 'max_video_duration_minutes_str': 'None'}
Traceback (most recent call last):
  File "csv_run_inference.py", line 113, in <module>
    inferenceEngine = edgetpu.detection.engine.BasicEngine(args['model'])
  File "/usr/local/lib/python3.7/dist-packages/edgetpu/basic/basic_engine.py", line 40, in __init__
    self._engine = BasicEnginePythonWrapper.CreateFromFile(model_path)
RuntimeError: No Edge TPU device detected!
```
