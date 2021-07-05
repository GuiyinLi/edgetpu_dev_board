import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
import glob

from distutils.version import StrictVersion
from collections import defaultdict
from io import StringIO
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from PIL import Image
import argparse

from pretrained_evaluate_utils import *

if StrictVersion(tf.__version__) < StrictVersion('1.12.0'):
  raise ImportError('Please upgrade your TensorFlow installation to v1.12.*.')

# This is needed since the notebook is stored in the object_detection folder.
ROOT_TF_MODELS_DIR=os.environ['TF_MODELS_DIR']
sys.path.append(ROOT_TF_MODELS_DIR)

OBJECT_DETECT_DIR=ROOT_TF_MODELS_DIR + '/object_detection/'
sys.path.append(OBJECT_DETECT_DIR)

from object_detection.utils import ops as utils_ops
from utils import label_map_util
from utils import visualization_utils as vis_util

parser = argparse.ArgumentParser(description='Evaluate Pre-trained object detector')
parser.add_argument('--input-images-dir', type=str)
parser.add_argument('--output-images-dir', type=str)
parser.add_argument('--model-name', type=str)
parser.add_argument('--frozen-graph-path', type=str)
parser.add_argument('--labels-path', type=str)
parser.add_argument('--max-images', type=int, default=None)

args = parser.parse_args()

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_FROZEN_GRAPH = args.frozen_graph_path
MODEL_NAME = args.model_name

print('PATH: ', PATH_TO_FROZEN_GRAPH)

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = args.labels_path
OUTPUT_IMAGES_DIR = args.output_images_dir

category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)


# For the sake of simplicity we will use only 2 images:
# image1.jpg
# image2.jpg
# If you want to test the code with your images, just add path to the images to the TEST_IMAGE_PATHS.
PATH_TO_TEST_IMAGES_DIR = args.input_images_dir
ALL_TEST_IMAGE_PATHS = glob.glob(PATH_TO_TEST_IMAGES_DIR + '/*.jpg')

max_images = args.max_images

if max_images > 0:
    TEST_IMAGE_PATHS = ALL_TEST_IMAGE_PATHS[0:max_images]
else:
    TEST_IMAGE_PATHS = ALL_TEST_IMAGE_PATHS

print(TEST_IMAGE_PATHS)

# Size, in inches, of the output images.
IMAGE_SIZE = (12, 8)

LINE_THICKNESS = 12

Detector = Classifier(PATH_TO_MODEL = PATH_TO_FROZEN_GRAPH)

all_rows = []

csv_fname = OUTPUT_IMAGES_DIR + '/' + '_'.join([MODEL_NAME, 'image', 'annotations']) + '.csv'  

for i, image_path in enumerate(TEST_IMAGE_PATHS):
  image = Image.open(image_path)
  # the array based representation of the image will be used later in order to prepare the
  # result image with boxes and labels on it.
  image_np = load_image_into_numpy_array(image)
  # Expand dimensions since the model expects images to have shape: [1, None, None, 3]

  output_dict = Detector.get_classification(img = image_np)

  # Visualization of the results of a detection.
  vis_util.visualize_boxes_and_labels_on_image_array(
      image_np,
      output_dict['detection_boxes'],
      output_dict['detection_classes'],
      output_dict['detection_scores'],
      category_index,
      instance_masks=output_dict.get('detection_masks'),
      use_normalized_coordinates=True,
      line_thickness=LINE_THICKNESS)
  plt.figure(figsize=IMAGE_SIZE)
  plt.imshow(image_np)
  plot_path = OUTPUT_IMAGES_DIR + '/' + '_'.join([MODEL_NAME, str(i)]) + '.png'  
  plt.savefig(plot_path)
  
  boxes = output_dict['detection_boxes']
  classes = output_dict['detection_classes']
  scores = output_dict['detection_scores']
  
  for j, box in enumerate(boxes):
      if scores[j] < 0.5: #0.0001:
          continue
      object_name = category_index[classes[j]]['name']
      ymin, xmin, ymax, xmax = boxes[j]
      ymin = ymin * image_np.shape[0]
      ymax = ymax * image_np.shape[0]
      xmin = xmin * image_np.shape[1]
      xmax = xmax * image_np.shape[1]
      row = [i, object_name, scores[j], xmin, ymin, xmax, ymax, image_path, MODEL_NAME]
      all_rows.append(row)

# create a dataframe with the correct format
df = get_df(all_rows)
df.to_csv(csv_fname, index = False)
