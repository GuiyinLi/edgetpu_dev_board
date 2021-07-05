import sys, os
import numpy as np
import cv2
import argparse

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
ap.add_argument("-n", "--video-num", type=str, required=False, default=1803)
ap.add_argument("-o", "--output-video-dir", required=False, help="path to video")
ap.add_argument("--model-name", required=False, help="path to video")
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
ap.add_argument("--csv-annotations", type=str, default='False')

args = vars(ap.parse_args())

# begin main here
####################
if  __name__ == '__main__':

    BASE_VIDEO_DIR = args['base_video_dir']
    VIDEO_NUM = args['video_num']
    OUTPUT_VIDEO_DIR = args['output_video_dir']

    print_mode_str = args['print_mode']

    if print_mode_str == 'False':
        print_mode = False
    else:
        print_mode = True

    # input video file
    video_file = BASE_VIDEO_DIR + '/' + str(VIDEO_NUM) + '.MP4'


    # Use Google Corals own DetectionEngine for handling
    # communication with the Coral
    inferenceEngine = edgetpu.detection.engine.DetectionEngine(args['model'])

    # Store labels for matching with inference results
    labels = ReadLabelFile(args['labels']) if args['labels'] else None

    # Specify font for labels
    #font = PIL.ImageFont.truetype("/usr/share/fonts/truetype/piboto/Piboto-Regular.ttf", 30)
    font = PIL.ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf", 35)


    #output_video_file = OUTPUT_VIDEO_DIR + '/output_' + str(VIDEO_NUM) + '.avi'
    video_name = '_'.join(['edge_TPU', str(args["model_name"]), str(VIDEO_NUM), 'confidence', str(args["confidence"]), 'max_objects', str(args["maxobjects"])])

    output_video_file = OUTPUT_VIDEO_DIR + '/' + video_name + '.avi'

    cap = cv2.VideoCapture(video_file)
    fw = int(cap.get(3))
    fh = int(cap.get(4))
    print("w,h",fw,fh)

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    fps = 30.0
    out = cv2.VideoWriter(output_video_file,fourcc, fps, (fw,fh))

    frame_number = 0
    MAX_FRAMES = numpy.inf

    all_rows = []
    inference_time = []
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret and frame_number <= MAX_FRAMES:
            # run original frame

            # Prepare screenshot for annotation by reading it into a PIL IMAGE object
            image = PIL.Image.fromarray( frame )

            # Perform inference and note time taken
            startMs = time.time()
            inferenceResults = inferenceEngine.DetectWithImage(image, threshold=args['confidence'], keep_aspect_ratio=True, relative_coord=False, top_k=args['maxobjects'])
            inference_time.append(inferenceEngine.get_inference_time())
            elapsedMs = time.time() - startMs

            #print(' ')
            #print('inference results: ', inferenceResults)
            #print('time: ', elapsedMs)
            #print(' ')

            annotated_frame, inference_list = annotate_and_display(image, inferenceResults, elapsedMs, labels, frame_number, font = font, print_mode = print_mode)

            for row in inference_list:
                all_rows.append(row)

            # write the unflipped frame
            out.write(annotated_frame)

            frame_number +=1
            if cv2.waitKey(1) & 0xFF == ord('q'):
                # now write the rows to a csv
                if args['csv_annotations'] != 'False':
                    write_csv(row_list = all_rows, csv_name = args['csv_annotations'])
                break

            if frame_number % 50 == 0:
                print('frame_number: ', frame_number)
                print(' ')
                #print('inference results: ', inferenceResults)
                print('time milliseconds: ', elapsedMs * 1000.0)
                print(' ')
        else:
            print('mean inference time: ', np.mean(inference_time))
            print('inf time std: ', np.std(inference_time))
            print('DONE VIDEO')

            # now write the rows to a csv
            if args['csv_annotations'] != 'False':
                write_csv(row_list = all_rows, csv_name = args['csv_annotations'])
            break


    # Release everything if job is finished
    cap.release()
    out.release()
    cv2.destroyAllWindows()


