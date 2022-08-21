#! /usr/bin/env python3

import os

# 카메라 모듈 불러오기
from camera import *

# 모터 
from scout_motor_light_pub import *

# key 변수로 모터 제어 함수 불러오기
from key_move import *

# RGB값을 이용한 주행 알고리즘 불러오기
from drive import *

# 깊이값 활용하는 코드
from utils2 import *
from Default_dist import *

import tensorflow as tf
from absl import app, flags
from absl.flags import FLAGS
from tensorflow.python.saved_model import tag_constants
# from core.config import cfg
import core.utils as utils
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.compat.v1 import ConfigProto

# deep sort imports
from deep_sort.detection import Detection
from deep_sort import nn_matching, preprocessing
from tools import generate_detections as gdet

# comment out below line to enable tensorflow logging outputs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0: tf.config.experimental.set_memory_growth(physical_devices[0], True)

flags.DEFINE_string('weights_marker', os.getenv('HOME') + '/wego_ws/src/scout_ros/scout_bringup/checkpoints/yolov4-tiny-416-marker', 'path to weights file')
flags.DEFINE_integer('size', 416, 'resize images to')
flags.DEFINE_float('iou', 0.45, 'iou threshold')
flags.DEFINE_float('score', 0.50, 'score threshold')
flags.DEFINE_boolean('count', False, 'count objects being tracked on screen')
flags.DEFINE_string('images_base_path', os.getenv('HOME') + '/Documents/exp1_test_dataset_1', 'path to images files') # 경로 설정하기
flags.DEFINE_string('marker_to_test', '10', 'the marker to test')

# bounding box 색깔 지정
cmap = plt.get_cmap('tab20b')
colors = [cmap(i)[:3] for i in np.linspace(0, 1, 6)]
marker_colors = [[i * 255 for i in color] for color in colors[:5]]

# detection indexing
idx_to_name_marker = {0: "0", 1: "10", 2: "20", 3: "30", 4: "40"}
name_to_idx_marker = {"0": 0, "10": 1, "20": 2, "30": 3, "40": 4}

def detect_objects(infer, batch_data, frame, encoder, idx_to_name):
    nms_max_overlap = 1.0

    pred_bbox = infer(batch_data)
    for _, value in pred_bbox.items():
        boxes = value[:, :, 0:4]
        pred_conf = value[:, :, 4:]

    boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
                boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
                scores=tf.reshape(
                    pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
                max_output_size_per_class=50,
                max_total_size=50,
                iou_threshold=FLAGS.iou,
                score_threshold=FLAGS.score
            )

    # convert data to numpy arrays and slice out unused elements
    num_objects = valid_detections.numpy()[0]
    bboxes = boxes.numpy()[0]
    bboxes = bboxes[0:int(num_objects)]
    scores = scores.numpy()[0]
    scores = scores[0:int(num_objects)]
    classes = classes.numpy()[0]
    classes = classes[0:int(num_objects)]

    # format bounding boxes from normalized ymin, xmin, ymax, xmax ---> xmin, ymin, width, height
    original_h, original_w, _ = frame.shape
    bboxes = utils.format_boxes(bboxes, original_h, original_w) 

    names = [idx_to_name[i] for i in classes]

    # encode yolo detections and feed to tracker
    features = encoder(frame, bboxes)
    detections = [Detection(bbox, score, class_name, feature) for bbox, score, class_name, feature in zip(bboxes, scores, names, features)]

    # run non-maxima supression
    boxs = np.array([d.tlwh for d in detections])
    scores = np.array([d.confidence for d in detections])
    classes = np.array([d.class_name for d in detections])
    indices = preprocessing.non_max_suppression(boxs, classes, nms_max_overlap, scores)
    detections = [detections[i] for i in indices]
    
    return detections

def test_marker_detection(infer_marker, encoder, *test_type):
    
    test_case = "test_" + "_".join(map(str, test_type))
    print("=============================================")
    print(f"[Test Case] '{test_case}' ---> Start")
    
    frame_num = 0 # 전체 영상 프레임 수
    count = 0 # 검출된 프레임 수

    marker_images_path = os.path.join(FLAGS.images_base_path, test_case)
    
    for image_file in os.listdir(marker_images_path):
        
        image_path = os.path.join(marker_images_path, image_file)
        
        frame = cv2.imread(image_path, cv2.IMREAD_COLOR)
        
        if frame is None:
            print("Image load failed")
            return
        
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # 프레임 넘버 1 증가
        frame_num +=1

        # 프레임 이미지 정보
        image_data = cv2.resize(frame, (FLAGS.size, FLAGS.size))
        image_data = image_data / 255.
        image_data = image_data[np.newaxis, ...].astype(np.float32)
        
        batch_data = tf.constant(image_data)
        
        detections_marker = detect_objects(infer_marker, batch_data, frame, encoder, idx_to_name_marker)
        # draw bbox for all detections
        detected = False
        for detection in detections_marker:
            name = detection.get_class()
            bbox = detection.to_tlbr()
            if name == FLAGS.marker_to_test:
                print(f"{image_file}: Detected")
                detected = True
                count += 1 # 존재 시 count 1 증가
                color = marker_colors[name_to_idx_marker[name]]
                cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
                cv2.rectangle(frame, (int(bbox[0]), int(bbox[1]-30)), (int(bbox[0])+34, int(bbox[1])), color, -1)
                cv2.putText(frame, name, (int(bbox[0]), int(bbox[1]-10)), 0, 0.75, (255,255,255), 2)
                break

            color = marker_colors[name_to_idx_marker[name]]
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1]-30)), (int(bbox[0])+34, int(bbox[1])), color, -1)
            cv2.putText(frame, name, (int(bbox[0]), int(bbox[1]-10)), 0, 0.75, (255,255,255), 2)

        if not detected: print(f"{image_file}: Undetected")
        
        result = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.imshow("Output Video", result)
        
        keyboard = cv2.waitKey(1) & 0xFF
        if keyboard == 27 or keyboard == 113: # ESC 또는 q 입력으로 프로그램 종료
            # dc.release()
            cv2.destroyAllWindows()
            print(f"key 'ESC' or 'q' ---> End")
            break
            
    result = count / frame_num
    print(f"[Test Case] '{test_case}' ---> Detection Rate: {result}")
    print(f"[Test Case] '{test_case}' ----> End")    
    print(f"=============================================")
        
def main(_argv):
    del _argv
    
    # initialize deep sort
    encoder = gdet.create_box_encoder(os.getenv('HOME') + '/wego_ws/src/scout_ros/scout_bringup/model_data/mars-small128.pb', batch_size=1)

    # load configuration for object detector
    config = ConfigProto()
    config.gpu_options.allow_growth = True

    saved_model_loaded_marker = tf.saved_model.load(FLAGS.weights_marker, tags=[tag_constants.SERVING])
    infer_marker = saved_model_loaded_marker.signatures['serving_default']
    
    print("Detection Rate Experiment - YOLO")
    for i in range(2): # 방향
        for j in range(3): # 간격
            test_marker_detection(infer_marker, encoder, i, j)
    test_marker_detection(infer_marker, encoder, 2)
    
if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass