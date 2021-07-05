import sys, os
import numpy as np
import cv2
import argparse
import csv

import edgetpu.detection.engine
from edgetpu.utils import image_processing

import time


from utils_run_video import *

waitkey_duration=10
display_video_mode = False
print_mode = True

# construct the argument parse and parse the arguments
####################
ap = argparse.ArgumentParser()
ap.add_argument("-b", "--base-video-dir", required=False, help="path to video")
ap.add_argument("-n", "--video_num", type=str, required=False, default=0)
ap.add_argument("-o", "--output-video-dir", required=False, help="path to video")
ap.add_argument("--model_name", required=False, help="path to video")
ap.add_argument( '--model', type=str,
                     default='mobilenet_ssd_v2_coco_quant_postprocess_edgetpu.tflite',
                     help="Path to the neural network graph file." )
ap.add_argument( '--labels', type=str,
                     default=None,
                     help="Path to labels file." )
ap.add_argument( '--maxobjects', type=int,
                     default=3,
                     help="Maximum objects to infer in each frame of video." )
ap.add_argument( '--confidence', type=float,
                     default=0.60,
                     help="Minimum confidence threshold to tag objects." )
ap.add_argument('--ct', required=True, help='using cpu or tpu for DNN inference')
ap.add_argument("-p", "--print-mode", type=str, default='False')
ap.add_argument("--out-video-create-mode", type=str, default='False', help='create an output video?')
ap.add_argument("--csv_annotations", required=True, help='where to write output annotations')
ap.add_argument("--use_webcam", default='False', help='true if we want use live webcam input')
ap.add_argument("--write_frame_with_predictions_str", default='True', help='true if output video should have bounding boxes overlaid, false is just raw video')
ap.add_argument("--max_video_duration_minutes_str", default='None', help='None means goes forever, else how many minutes to record video for')

args = vars(ap.parse_args())

# begin main here
####################
if  __name__ == '__main__':

    BASE_VIDEO_DIR = args['base_video_dir']
    VIDEO_NUM = args['video_num']
    OUTPUT_VIDEO_DIR = args['output_video_dir']
    print_mode_str = args['print_mode']
    out_video_create_mode_str = args['out_video_create_mode']
    use_webcam_str = args['use_webcam']
    write_frame_with_predictions_str = args['write_frame_with_predictions_str']
    max_video_duration_minutes_str = args['max_video_duration_minutes_str']

    print('args: ', args)

    if print_mode_str == 'False':
        print_mode = False
    else:
        print_mode = True

    if use_webcam_str == 'False':
        use_webcam = False
        # input video file
        #video_file = BASE_VIDEO_DIR + '/' + str(VIDEO_NUM) + '.MP4'
        video_file = BASE_VIDEO_DIR + '/' + str(VIDEO_NUM) + '.avi'
    else:
        use_webcam = True

    # whether to create an output video
    if out_video_create_mode_str == 'False':
        out_video_create_mode = False
    else:
        out_video_create_mode = True

    # true if the output video should have annotations overlaid
    if write_frame_with_predictions_str == 'False':
        write_frame_with_predictions = False
    else:
        write_frame_with_predictions = True

    if max_video_duration_minutes_str == 'None':
        # don't stop recording after max frames
        MAX_FRAMES = np.inf
    else:
        # approximately
        fps = 30.0
        seconds_per_min = 60
        MAX_FRAMES = int(fps * seconds_per_min * int(max_video_duration_minutes_str))
        print('MAX_FRAMES to capture: ', MAX_FRAMES)
        print('max minutes: ', max_video_duration_minutes_str)


    # Use Google Corals own DetectionEngine for handling
    # communication with the Coral
    inferenceEngine = edgetpu.detection.engine.DetectionEngine(args['model'])

    # Store labels for matching with inference results
    labels = ReadLabelFile(args['labels']) if args['labels'] else None

    # Specify font for labels
    font = PIL.ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf", 35)

    # what the name the output video
    video_name = '_'.join(['edge_TPU', str(args["model_name"]), str(VIDEO_NUM), 'confidence', str(args["confidence"]), 'max_objects', str(args["maxobjects"]), 'has_annotations', str(write_frame_with_predictions_str)])

    # MP4
    output_video_file = OUTPUT_VIDEO_DIR + '/' + video_name + '.mp4'

    # for avi case
    # output_video_file = OUTPUT_VIDEO_DIR + '/' + video_name + '.avi'

    # webcam is connected
    if use_webcam:
        # just for testing
        cap = cv2.VideoCapture(1)
        print('USING WEBCAM')
    else:
        cap = cv2.VideoCapture(video_file)
        print('USING stored video: ', video_file)

    # display image dimensions
    fw = int(cap.get(3))
    fh = int(cap.get(4))
    print("w,h",fw,fh)


    if out_video_create_mode:
        # Define the codec and create VideoWriter object
        # for AVI files
        #fourcc = cv2.VideoWriter_fourcc(*'XVID')
        # for MP4 files, but has warning on edge TPU
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = 30.0
        out = cv2.VideoWriter(output_video_file,fourcc, fps, (fw,fh))

    frame_number = 0

    csv_file = args['csv_annotations'] + '/' + str(VIDEO_NUM) + '_' + args['model_name'] + '_' + args['ct'] + '.csv'
    all_rows = []
    header_row = ['frame', 'cname', 'conf', 'xmin', 'ymin', 'xmax', 'ymax', 'time_ms']
    all_rows.append(header_row)
    inference_t_l = []
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret and frame_number <= MAX_FRAMES:
            # run original frame

            # Prepare screenshot for annotation by reading it into a PIL IMAGE object
            image = PIL.Image.fromarray( frame )

            # Perform inference and note time taken
            startMs = time.time()

	    # for dev board
            inferenceResults = inferenceEngine.DetectWithImage(image, threshold=args['confidence'], keep_aspect_ratio=True, relative_coord=False, top_k=args['maxobjects'])
            elapsedMs = time.time() - startMs
            inference_time = inferenceEngine.get_inference_time()
            #print(' ')
            #print('inference results: ', inferenceResults)
            #print('time: ', elapsedMs)
            #print(' ')
            inference_t_l.append(inference_time)
            annotated_frame, inference_list = annotate_and_display(image, inferenceResults, elapsedMs, labels, frame_number, font = font, print_mode = print_mode, inference_time_ms = inference_time)

            for row in inference_list:
                all_rows.append(row)

            if out_video_create_mode:
                # write the unflipped frame
                if write_frame_with_predictions:
                    out.write(annotated_frame)
                else:
                    out.write(frame)

            frame_number +=1
            if cv2.waitKey(1) & 0xFF == ord('q'):
                # now write the rows to a csv
                write_csv(csv_name = csv_file, row_list = all_rows)
                print('Average inference time over each frame of ' + args['video_num'] + ' with ' + args['model_name'] +  ' in ' + str(args['ct']) + ' is ' +str(np.mean(inference_t_l)) + 'ms')
                print('The std is ' + str(np.std(inference_t_l)))
                break

            if frame_number % 50 == 0:
                print('frame_number: ', frame_number)
                print('time ms: ', elapsedMs * 1000.0)
        else:
            print('DONE VIDEO')

            # now write the rows to a csv
            write_csv(csv_name = csv_file, row_list = all_rows)
            print('Average inference time over each frame of ' + args['video_num'] + ' with ' + args['model_name'] + ' in ' + str(args['ct']) + ' is ' +str(np.mean(inference_t_l)) +    'ms')
            print('The std is ' + str(np.std(inference_t_l)))
            break

    # Release everything if job is finished
    cap.release()

    if out_video_create_mode:
        out.release()
    cv2.destroyAllWindows()

