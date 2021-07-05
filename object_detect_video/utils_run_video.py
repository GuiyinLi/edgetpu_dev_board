import os
import cv2
import sys
import csv

import PIL.Image
import PIL.ImageDraw
import PIL.ImageFont
import numpy

def annotate_single_image ( image, inferenceResults, elapsedMs, labels, frame_number, image_path, font = None, print_mode = False, inference_time_ms = None):
    # Iterate through result list. Note that results are already sorted by
    # confidence score (highest to lowest) and records with a lower score
    # than the threshold are already removed.
    result_size = len(inferenceResults)

    per_object_results = []


    for idx, obj in enumerate(inferenceResults):

        # Prepare image for drawing
        draw = PIL.ImageDraw.Draw( image )

        # Prepare boundary box
        box = obj.bounding_box.flatten().tolist()
        #print('box: ', obj.bounding_box)

        # according to google docs: 0,0 is top left
        # x1, y1 (TOP LEFT box), x2, y2 (BOTTOM RIGHT box)
        # so x1 = xmin, y1 = ymin (since (0,0) is top LEFT)
        # x2 = xmax and y2 = ymax (again since (0,0) is top LEFT)

        #xmin = x1
        xmin = box[0]
        # ymax = y1
        ymin = box[1]

        #xmax = x2
        xmax = box[2]
        #ymin = y2
        ymax = box[3]

        print('xmin: ', xmin)
        print('xmax: ', xmax)
        print('ymin: ', ymin)
        print('ymax: ', ymax)

        # Draw rectangle to desired thickness
        for x in range( 0, 4 ):
            draw.rectangle(box, outline=(255, 255, 0))

        # Annotate image with label and confidence score
        display_str = labels[obj.label_id] + ": " + str(round(obj.score*100, 2)) + "%"
        draw.text( (box[0], box[1]), display_str, font = font)
        if print_mode:
            print("Object (" + str(idx+1) + " of " + str(result_size) + "): "
                  + labels[obj.label_id] + " (" + str(obj.label_id) + ")"
                  + ", Confidence:" + str(obj.score)
                  + ", Elapsed:" + str(elapsedMs*1000.0) + "ms"
                  + ", Box:" + str(box))


        per_object_results.append([frame_number, labels[obj.label_id], obj.score, xmin, ymin, xmax, ymax, image_path, inference_time_ms])

    # If a display is available, show the image on which inference was performed
    displayImage = numpy.asarray( image )
    if 'DISPLAY' in os.environ:
        cv2.imshow( 'NCS Improved live inference', displayImage )


    return displayImage, per_object_results


# Read labels from text files.
def ReadLabelFile(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
        ret = {}
    for line in lines:
        pair = line.strip().split(maxsplit=1)
        ret[int(pair[0])] = pair[1].strip()
    return ret

# Annotate and display video
def annotate_and_display ( image, inferenceResults, elapsedMs, labels, frame_number, font = None, print_mode = False, inference_time_ms = None):
    # Iterate through result list. Note that results are already sorted by
    # confidence score (highest to lowest) and records with a lower score
    # than the threshold are already removed.
    result_size = len(inferenceResults)

    per_object_results = []


    for idx, obj in enumerate(inferenceResults):

        # Prepare image for drawing
        draw = PIL.ImageDraw.Draw( image )

        # Prepare boundary box
        box = obj.bounding_box.flatten().tolist()

        # x1, y1 (TOP LEFT), x2, y2 (BOTTOM RIGHT)
        #xmin = x1
        xmin = box[0]
        # ymax = y1
        ymin = box[1]

        #xmax = x2
        xmax = box[2]
        #ymin = x3
        ymax = box[3]

        # Draw rectangle to desired thickness
        for x in range( 0, 4 ):
            draw.rectangle(box, outline=(255, 255, 0))

        # Annotate image with label and confidence score
        if labels:
            display_str = labels[obj.label_id] + ": " + str(round(obj.score*100, 2)) + "%"
        else:
            display_str = str(round(obj.score*100, 2)) + "%"

        draw.text( (box[0], box[1]), display_str, font = font)
        if print_mode:
            print("Object (" + str(idx+1) + " of " + str(result_size) + "): "
                  + labels[obj.label_id] + " (" + str(obj.label_id) + ")"
                  + ", Confidence:" + str(obj.score)
                  + ", Elapsed:" + str(elapsedMs*1000.0) + "ms"
                  + ", Box:" + str(box))


        if labels:
            per_object_results.append([frame_number, labels[obj.label_id], obj.score, xmin, ymin, xmax, ymax, inference_time_ms])
        else:
            per_object_results.append([frame_number, '', obj.score, xmin, ymin, xmax, ymax, inference_time_ms])

    # If a display is available, show the image on which inference was performed
    displayImage = numpy.asarray( image )
    if 'DISPLAY' in os.environ:
        cv2.imshow( 'NCS Improved live inference', displayImage )


    return displayImage, per_object_results


def get_df(all_rows):
    df = pd.DataFrame(all_rows,
                      columns=['frame', 'cname', 'conf', 'xmin', 'ymin', 'xmax', 'ymax'])
    f32 = ['conf', 'xmin', 'ymin', 'xmax', 'ymax']
    for f in f32:
        df[f] = df[f].astype('float32')
    df['frame'] = df['frame'].astype('int32')
    df['cname'] = df['cname'].astype(str)
    df = df.sort_values(by=['frame', 'cname', 'conf'], ascending=[True, True, False])
    return df

def write_csv(row_list = None, csv_name = None):
    with open(csv_name, 'w') as resultFile:
        wr = csv.writer(resultFile, dialect='excel')
        wr.writerows(row_list)

