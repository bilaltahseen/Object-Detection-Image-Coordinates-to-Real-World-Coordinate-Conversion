# import the opencv library
import cv2
from PIL import Image, ImageDraw
import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
import pathlib
import math
from collections import defaultdict
from io import StringIO
import time

from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

# %%
# patch tf1 into `utils.ops`
utils_ops.tf = tf.compat.v1

# Patch the location of gfile
tf.gfile = tf.io.gfile


# %%
# Loader
# %%
def load_model(model_dir):
    # base_url = 'http://download.tensorflow.org/models/object_detection/'
    # model_file = model_name + '.tar.gz'
    # model_dir = tf.keras.utils.get_file(
    #     fname=model_name,
    #     origin=base_url + model_file,
    #     untar=True)
    model_dir = pathlib.Path(model_dir)/"saved_model"
    model = tf.saved_model.load(str(model_dir))
    return model


# %%
# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = './fine_tuned_model_ssd/Metals_label_map.pbtxt'
category_index = label_map_util.create_category_index_from_labelmap(
    PATH_TO_LABELS, use_display_name=True)


# %%
model_dir = './fine_tuned_model_ssd'
detection_model = load_model(model_dir)
print(detection_model.signatures['serving_default'].inputs)
detection_model.signatures['serving_default'].output_dtypes
detection_model.signatures['serving_default'].output_shapes


def run_inference_for_single_image(model, image):
    image = np.asarray(image)
    # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
    input_tensor = tf.convert_to_tensor(image)
    # The model expects a batch of images, so add an axis with `tf.newaxis`.
    input_tensor = input_tensor[tf.newaxis, ...]

    # Run inference
    model_fn = model.signatures['serving_default']
    output_dict = model_fn(input_tensor)

    # All outputs are batches tensors.
    # Convert to numpy arrays, and take index [0] to remove the batch dimension.
    # We're only interested in the first num_detections.
    num_detections = int(output_dict.pop('num_detections'))
    output_dict = {key: value[0, :num_detections].numpy()
                   for key, value in output_dict.items()}
    output_dict['num_detections'] = num_detections

    # detection_classes should be ints.
    output_dict['detection_classes'] = output_dict['detection_classes'].astype(
        np.int64)

    # Handle models with masks:
    if 'detection_masks' in output_dict:
        # Reframe the the bbox mask to the image size.
        detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
            output_dict['detection_masks'], output_dict['detection_boxes'],
            image.shape[0], image.shape[1])
        detection_masks_reframed = tf.cast(detection_masks_reframed > 0.5,
                                           tf.uint8)
        output_dict['detection_masks_reframed'] = detection_masks_reframed.numpy()

    return output_dict


def show_inference(model, image_path):
    objects_dict = []
    # the array based representation of the image will be used later in order to prepare the
    # result image with boxes and labels on it.
    image_np = np.array(Image.open(image_path))
    # Actual detection.
    output_dict = run_inference_for_single_image(model, image_np)
    # Visualization of the results of a detection.
    vis_util.visualize_boxes_and_labels_on_image_array(
        image_np,
        output_dict['detection_boxes'],
        output_dict['detection_classes'],
        output_dict['detection_scores'],
        category_index,
        instance_masks=output_dict.get('detection_masks_reframed', None),
        use_normalized_coordinates=True,
        min_score_thresh=0.8,
        line_thickness=8)

    boxes = np.squeeze(output_dict['detection_boxes'])
    scores = np.squeeze(output_dict['detection_scores'])
    classes = np.squeeze((output_dict['detection_classes']).astype(int))

    min_score_thresh = 0.8

    bboxes = boxes[scores >= min_score_thresh]
    cclasses = classes[scores >= min_score_thresh]
    sscores = scores[scores >= min_score_thresh]
    IMAGE = Image.fromarray(image_np)
    draw = ImageDraw.Draw(IMAGE)
    im_width = IMAGE.width
    im_height = IMAGE.height

    for box, classes, scores in zip(bboxes, cclasses, sscores):
        ymin, xmin, ymax, xmax = box
        (left, right, top, bottom) = (xmin * im_width,
                                      xmax * im_width, ymin * im_height, ymax * im_height)
        left, right, top, bottom = np.int(left), np.int(
            right), np.int(top), np.int(bottom)

        a0 = abs(right-(right-left)*0.5)
        b0 = abs(bottom)
        a1 = abs(right-(right-left)*0.5)
        b1 = abs(im_height)
        x1, y1 = a0, b0
        x2, y2 = a1, b1
        x1, y1, x2, y2 = np.int(x1), np.int(y1), np.int(x2), np.int(y2)
        ob_center_x0, ob_center_y0 = abs(
            right-(right-left)*0.5), abs(bottom-(bottom-top)*0.5)
        draw.text((ob_center_x0, ob_center_y0), "{},{}".format(
            ob_center_x0, ob_center_y0), fill='yellow')
        draw.line((0, ob_center_y0, im_width, ob_center_y0), fill='red')
        draw.line((ob_center_x0, 0, ob_center_x0, im_height), fill='red')
        objects_dict.append({'name': category_index[classes]['name'], 'pixel_cord': (
            ob_center_x0, ob_center_y0)})
    IMAGE.show()
    return objects_dict


# define a video capture object
cap = cv2.VideoCapture(1)


def camera_cord_to_manu_cord(id, x_loc, y_loc):
    Rad = (95.0/180.0) * np.pi
    RZ = [[np.cos(Rad), -np.sin(Rad), 0],
          [np.sin(Rad), np.cos(Rad), 0], [0, 0, 1]]
    R180_X = [[1, 0, 0], [
        0, np.cos(np.pi), -np.sin(np.pi)], [0, np.sin(np.pi), np.cos(np.pi)]]
# R90_X = [[np.cos(np.pi/2), -np.sin(np.pi/2), 0],
#          [np.sin(np.pi/2), np.cos(np.pi/2), 0], [0, 0, 1]]
    RO_C = np.dot(RZ, R180_X)
    # RO_C = R180_X

    DO_C = [[-8.5], [-22], [0]]

    HO_C = np.concatenate((RO_C, DO_C), 1)
    HO_C = np.concatenate((HO_C, [[0, 0, 0, 1]]), 0)
    # px_to_cm_ration = 19.1/640.0
    px_to_cm_ration = 22/640.0
    X_C = x_loc*px_to_cm_ration
    Y_C = y_loc*px_to_cm_ration
    PC = [[X_C], [Y_C], [0], [1]]
    PO = np.dot(HO_C, PC)
    print('ID', id)
    print('WRT_MANUPLATOR', PO[0], PO[1])
    print('WRT_CAMERA_CENTER', X_C, Y_C)


while(True):
    # Capture the video frame
    # by frame
    ret, frame = cap.read()
    cv2.circle(frame, (0, 0), radius=5, color=(0, 0, 255), thickness=-1)

    cv2.imshow('frame', frame)
    # print(frame.shape)
    # the 'q' button is set as the
    # quitting button you may use any
    # desired button of your choice
    if cv2.waitKey(33) & 0xFF == ord('q'):
        break
    if cv2.waitKey(33) == ord('a'):
        cv2.imwrite('image.jpg', frame)
        object_a_dict = show_inference(
            image_path='./image.jpg', model=detection_model)
        print(object_a_dict)
        for obj in object_a_dict:
            camera_cord_to_manu_cord(
                obj['name'], obj['pixel_cord'][0], obj['pixel_cord'][1])

        # After the loop release the cap object
cap.release()
# Destroy all the windows
cv2.destroyAllWindows()
