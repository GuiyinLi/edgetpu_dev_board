import sys, os
import numpy as np
import argparse
import csv
from PIL import Image
import glob

import edgetpu.detection.engine
from edgetpu.utils import image_processing

import time

TPU_ROOT_DIR=os.environ['TPU_CODE_DIR']
sys.path.append(TPU_ROOT_DIR + '/object_detect_video/')

from utils_run_video import *

print_mode = True

# construct the argument parse and parse the arguments
####################
ap = argparse.ArgumentParser()
ap.add_argument("-b", "--base-images-dir", required=False, help="path to video")
ap.add_argument("-o", "--output-images-dir", required=False, help="path to video")
ap.add_argument("--model_name", required=False, help="path to video")
ap.add_argument("--max_images", type=int, default=False)
ap.add_argument( '--model', type=str,
                     default='mobilenet_ssd_v2_coco_quant_postprocess_edgetpu.tflite',
                     help="Path to the neural network graph file." )
ap.add_argument( '--labels', type=str,
                     default='coco_labels.txt',
                     help="Path to labels file." )
ap.add_argument( '--maxobjects', type=int,
                     default=3,
                     help="Maximum objects to infer in each frame of video." )
ap.add_argument( '--confidence', type=float,
                     default=0.60,
                     help="Minimum confidence threshold to tag objects." )
ap.add_argument("-p", "--print-mode", type=str, default='False')

args = vars(ap.parse_args())

def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)


# begin main here
####################
if  __name__ == '__main__':

    base_images_dir = args['base_images_dir']
    output_images_dir = args['output_images_dir']
    print_mode_str = args['print_mode']
    MODEL_NAME = args['model_name']

    print('args: ', args)

    if print_mode_str == 'False':
        print_mode = False
    else:
        print_mode = True

    # Use Google Corals own DetectionEngine for handling
    # communication with the Coral
    inferenceEngine = edgetpu.detection.engine.DetectionEngine(args['model'])

    # Store labels for matching with inference results
    labels = ReadLabelFile(args['labels']) if args['labels'] else None

    # Specify font for labels
    font = PIL.ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf", 35)

    #font = PIL.ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf", 35)


    # For the sake of simplicity we will use only 2 images:
    # If you want to test the code with your images, just add path to the images to the TEST_IMAGE_PATHS.
    PATH_TO_TEST_IMAGES_DIR = base_images_dir
    ALL_TEST_IMAGE_PATHS = glob.glob(PATH_TO_TEST_IMAGES_DIR + '/*.jpg')

    max_images = args["max_images"]

    if max_images > 0:
        TEST_IMAGE_PATHS = ALL_TEST_IMAGE_PATHS[0:max_images]
    else:
        TEST_IMAGE_PATHS = ALL_TEST_IMAGE_PATHS

    print(TEST_IMAGE_PATHS)

    # output csv
    csv_fname = output_images_dir + '/' + '_'.join([MODEL_NAME, 'image', 'annotations']) + '.csv'
    all_rows = []

    # old version
    header_row = ['frame', 'cname', 'conf', 'xmin', 'ymin', 'xmax', 'ymax', 'image_path', 'time_ms']

    all_rows.append(header_row)

    # keeps the inference times for std calc
    inference_t_l = []

    for i, image_path in enumerate(TEST_IMAGE_PATHS):
      image = Image.open(image_path)
      # the array based representation of the image will be used later in order to prepare the
      # result image with boxes and labels on it.
      #image_np = load_image_into_numpy_array(image)
      # Expand dimensions since the model expects images to have shape: [1, None, None, 3]


      # Perform inference and note time taken
      startMs = time.time()
      inferenceResults = inferenceEngine.DetectWithImage(image, threshold=args['confidence'], keep_aspect_ratio=True, relative_coord=False, top_k=args['maxobjects'])
      elapsedMs = time.time() - startMs
      inference_time = inferenceEngine.get_inference_time()
      inference_t_l.append(inference_time)

      #print('font', font)
      annotated_frame, inference_list = annotate_single_image(image, inferenceResults, elapsedMs, labels, i, image_path, font = font, print_mode = print_mode, inference_time_ms = inference_time)

      plot_path = output_images_dir + '/' + '_'.join([MODEL_NAME, str(i)]) + '.png'
      # WRITE OUTPUT IMAGE USING OPENCV, change this now!
      cv2.imwrite(plot_path, cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR))

      # keep
      for row in inference_list:
          #print('row: ', row)
          all_rows.append(row)

    # write this all to a csv now
    # now write the rows to a csv
    write_csv(csv_name = csv_fname, row_list = all_rows)
    print(' ')
    print('Average inference time over each frame of  with ' + args['model_name'] +  ' is ' +str(np.mean(inference_t_l)) + 'ms')
    print('The std is ' + str(np.std(inference_t_l)))
    print(' ')
