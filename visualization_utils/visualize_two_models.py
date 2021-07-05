# code to take a raw video and csv of annotations produced by 2 NN models
# and generate an ANNOTATED VIDEO that aggregates both models.

import argparse
import cv2
import pandas as pd
import numpy as np
import heapq

FPS = 30

object_to_color_map = {'car': (0x88, 0xCC, 0xEE),
                       'truck': (0xFF, 0x00, 0x99),
                       'person': (0x11, 0x77, 0x33),
                       'motorcycle': (0x99, 0x99, 0x33),
                       'bus': (0xDD, 0xCC, 0x77),
                       'bicycle': (0xFF, 0x00, 0x00),
                       'other': (0xDD, 0xCC, 0x77)}

def calc_intersection_over_union(label1, label2):
    # https://www.pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/
    boxA = [int(label1[3]), int(label1[4]),
            int(label1[5]), int(label1[6])]
    boxB = [int(label2[3]), int(label2[4]),
            int(label2[5]), int(label2[6])]

    # intersection box bounds
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    unionArea = float(boxAArea + boxBArea - interArea)

    iou = interArea / unionArea
    return iou

def draw_dashed_line(image, start, end, color, thickness, gap=10):
    # https://stackoverflow.com/questions/26690932/opencv-rectangle-with-dotted-or-dashed-lines/26711359#26711359
    dist = ((start[0] - end[0])**2 + (start[1] - end[1])**2)**.5
    points = []
    for i in np.arange(0, dist, gap):
        r = i / dist
        x = int((start[0] * (1-r) + end[0] * r) + .5)
        y = int((start[1] * (1-r) + end[1] * r) + .5)
        p = (x, y)
        points.append(p)

    s = points[0]
    e = points[0]
    i = 0
    for p in points:
        s = e
        e = p
        if i % 2 == 1:
            cv2.line(image, s, e, color, thickness)
        i += 1


# draws a single bounding box
def draw_label(label, image, color=(0, 255, 0), style='default', format='series'):
    if format == 'series':
        tl = (int(label.xmin), int(label.ymin))
        br = (int(label.xmax), int(label.ymax))
    else:
        tl = (int(label[3]), int(label[4]))
        br = (int(label[5]), int(label[6]))

    if style == 'dashed':
        points = [tl, (br[0], tl[1]), br, (tl[0], br[1])]
        start = points[0]
        end = points[0]
        points.append(points.pop(0))
        for p in points:
            start = end
            end = p
            draw_dashed_line(image, start, end, color, 2)
    else:
        cv2.rectangle(image, tl, br, color, 2)

    if format == 'series':
        text = '%s %2.1f' % (label.cname, label.conf * 100)
    else:
        text = '%s %2.1f' % (label[1], label[2] * 100)

    cv2.putText(image, text, tl, cv2.FONT_HERSHEY_SIMPLEX,
                0.75, (255, 255, 255), 2)

# draws bouding boxes for every video frame
def draw_video(video_path, output_video, diff_frame_dir, df1, df2, start_frame=0, num_frames=60, threshold=0.5, N=0, frame_count=5):
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    fourcc = int(cv2.VideoWriter_fourcc(*'mp4v'))
    frame_size = frame.shape[0:2][::-1]
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    writer = cv2.VideoWriter(output_video, fourcc, fps, frame_size)

    # min heap to store/cache the X most different frames
    frame_heap = []

    curr_frame = start_frame
    while ret and curr_frame < start_frame + num_frames:
        frame_label_1 = df1.loc[df1['frame'] == curr_frame]
        frame_label_2 = df2.loc[df2['frame'] == curr_frame]

        frame_objects = {}
        fnn_count = 0
        mn2_count = 0

        # objects detected by the FRCNN
        for _, row in frame_label_1.iterrows():
            print('row: ', row)
            # draw_label(row, frame, (0x11, 0x77, 0x33))
            frame_objects[tuple(row.values)] = 'frcnn' # pandas series is mutable, so conversion to tuple is necessary for hashing
            fnn_count += 1
        # objects detected by the MN2
        for _, row in frame_label_2.iterrows():
            print('row: ', row)
            # draw_label(row, frame, (0xFF, 0x00, 0x00), 'dashed')
            frame_objects[tuple(row.values)] = 'mn2'
            mn2_count += 1

        # finding which objects were similarly identified by both models
        frame_keys = list(frame_objects.keys())
        for i in range(len(frame_keys)):
            for j in range(i + 1, len(frame_keys)):
                objectA = frame_keys[i]
                objectB = frame_keys[j]
                iou = calc_intersection_over_union(objectA, objectB)
                # common object only if the 2 objects' IOU > threshold, their cnames match,
                # and they were identified by different NN models.
                if (iou > threshold and objectA[1] == objectB[1]
                    and frame_objects[objectA] != frame_objects[objectB]):
                    del frame_objects[objectA]
                    frame_objects[objectB] = 'common'

        # draw the bounding boxes
        for object in frame_objects.keys():
            if frame_objects[object] == 'frcnn':
                draw_label(object, frame, (0x11, 0x77, 0x33), 'default', 'tuple')
            elif frame_objects[object] == 'mn2':
                draw_label(object, frame, (0xFF, 0x00, 0x00), 'dashed', 'tuple')
            else:
                draw_label(object, frame, (0xFF, 0x00, 0x99), 'default', 'tuple')

        # keeping track of the difference in # of object detected per frame
        if mn2_count >= N and fnn_count >= N:
            diff_count = abs(fnn_count - mn2_count) # difference in # objects detected between the 2 models
            if len(frame_heap) < frame_count:
                heapq.heappush(frame_heap, (diff_count, curr_frame, frame))
            else:
                min_frame_tuple = heapq.heappop(frame_heap)
                # if the current frame has more differences than the frame with the fewest differences within the X frame heap/cache
                if min_frame_tuple[0] < diff_count:
                    heapq.heappush(frame_heap, (diff_count, curr_frame, frame))
                else:
                    heapq.heappush(frame_heap, min_frame_tuple)


        writer.write(frame)
        curr_frame += 1
        ret, frame = cap.read()

    cap.release()
    writer.release()

    # post processing to write 5 most different images to disk
    i = 1
    frame_cap = cv2.VideoCapture(video_path)
    for item in frame_heap:
        cv2.imwrite(diff_frame_dir + 'annotated_different_image_' + str(i) + '.jpg', item[2])
        frame_cap.set(1, item[1])
        ret, f = frame_cap.read()
        cv2.imwrite(diff_frame_dir + 'different_image_' + str(i) + '.jpg', f)
        i += 1
    frame_cap.release()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_csv_1', required=True, help='first CSV with classes and boxes')
    parser.add_argument('--input_csv_2', required=True, help='second CSV with classes and boxes')
    parser.add_argument('--video_path', required=True, help='the path of the video')
    parser.add_argument('--output_video', required=True, help='output video that is saved with the boxes')
    parser.add_argument('--diff_frame_dir', required=True, help='dir to contain X most different images')
    parser.add_argument('--start_frame', required=False, type=int, default=0, help='frame number from which the labeling starts')
    parser.add_argument('--num_frames', required=False, type=int, default=np.inf, help='number of frames to label')
    parser.add_argument('--iou_threshold', required=False, type=float, default=0.5, help='intersection over union threshold')
    parser.add_argument('--min_detections', required=False, type=int, default=0, help='minimim number of detections per model for frame diff analysis')
    parser.add_argument('--frame_count', required=False, type=int, default=5, help='number of different frames')


    args = parser.parse_args()

    df1 = pd.read_csv(args.input_csv_1)
    df2 = pd.read_csv(args.input_csv_2)

    draw_video(args.video_path, args.output_video, args.diff_frame_dir, df1, df2, args.start_frame, args.num_frames, args.iou_threshold, args.min_detections, args.frame_count)
