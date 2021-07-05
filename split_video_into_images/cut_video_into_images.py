import sys, os
import numpy as np
import cv2
import argparse

waitkey_duration=10
display_video_mode = False
print_mode = True

# construct the argument parse and parse the arguments
####################
ap = argparse.ArgumentParser()
ap.add_argument("-b", "--base-video-dir", required=False, help="path to video")
ap.add_argument("-n", "--video-num", type=str, required=False, default=1803)
ap.add_argument("-f", "--frame-skip", type=int, required=False, default=50, help="how many frames to skip")
args = vars(ap.parse_args())

# begin main here
####################

if  __name__ == '__main__':


    BASE_VIDEO_DIR = args['base_video_dir']
    VIDEO_NUM = args['video_num']
    FRAME_NUMBER = args['frame_skip']

    FRAME_DIR = BASE_VIDEO_DIR + '/' + str(VIDEO_NUM) + '_frames/'

    video_file = BASE_VIDEO_DIR + '/' + str(VIDEO_NUM) + '.MP4'

    cap = cv2.VideoCapture(video_file)

    count = 0
    while(cap.isOpened()):
        ret, frame = cap.read()

        if ret:

            if display_video_mode:
                cv2.imshow('Video', frame)

            if count % FRAME_NUMBER == 0:
                cv2.imwrite(FRAME_DIR + "/frame%d.jpg" % count, frame)

                if print_mode:
                    print('wrote frame ' + str(count) + '\n')

        else:
            print('invalid frame')
            break

        count+=1

        if cv2.waitKey(waitkey_duration) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

