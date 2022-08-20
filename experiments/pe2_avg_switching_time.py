#! /usr/bin/env python3

""" 타깃 마커 switching 후 타깃 사람 설정까지 걸리는 평균 시간 측정
마커 종류 5가지
경우의 수: 5C2 = 10
경우의 수 당 반복 횟수: 10
"""

import os

# 카메라 모듈 불러오기
from camera import *

# 모터 
from scout_motor_light_pub import *

# key 변수로 모터 제어 함수 불러오기
from key_move import *

# RGB값을 이용한 주행 알고리즘 불러오기
from drive import *

# 깊이값 활용하는 코드#! /usr/bin/env python3

""" 타깃 마커 switching 후 타깃 사람 설정까지 걸리는 평균 시간 측정
마커 종류 5가지
경우의 수: 5C2 = 10
경우의 수 당 반복 횟수: 10
"""

import os

# 카메라 모듈 불러오기
from camera import *

# 모터 
from scout_motor_light_pub import *

# key 변수로 모터 제어 함수 불러오기
from key_move import *

# RGB값을 이용한 주행 알고리즘 불러오기
from drive import *

import tensorflow as tf
from absl import app, flags
from absl.flags import FLAGS
from tensorflow.python.saved_model import tag_constants

# from core.config import cfg
import core.utils as utils
from PIL import Image
import cv2, math, time
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.compat.v1 import ConfigProto

# deep sort imports
from deep_sort.detection import Detection
from deep_sort import nn_matching, preprocessing
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet

# comment out below line to enable tensorflow logging outputs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0: tf.config.experimental.set_memory_growth(physical_devices[0], True)

flags.DEFINE_string('weights_person', os.getenv('HOME') + '/wego_ws/src/scout_ros/scout_bringup/checkpoints/yolov4-tiny-416-person', 'path to weights file')
flags.DEFINE_string('weights_marker', os.getenv('HOME') + '/wego_ws/src/scout_ros/scout_bringup/checkpoints/yolov4-tiny-416-marker', 'path to weights file')
flags.DEFINE_integer('size', 416, 'resize images to')
flags.DEFINE_float('iou', 0.45, 'iou threshold')
flags.DEFINE_float('score', 0.50, 'score threshold')
flags.DEFINE_boolean('count', False, 'count objects being tracked on screen')
flags.DEFINE_string('marker1', "0", 'marker 1')
flags.DEFINE_string('marker2', "10", 'marker 2')

# bounding box 색깔 지정
cmap = plt.get_cmap('tab20b')
colors = [cmap(i)[:3] for i in np.linspace(0, 1, 6)]
marker_colors = [[i * 255 for i in color] for color in colors[:5]]
person_color = [i * 255 for i in colors[5]]

# detection indexing
idx_to_name_marker = {0: "0", 1: "10", 2: "20", 3: "30", 4: "40"}
name_to_idx_marker = {"0": 0, "10": 1, "20": 2, "30": 3, "40": 4}
idx_to_name_person = {0: "person"}

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

def main(_argv):
    del _argv
    # target 설정을 위한 변수
    target_marker = "0"
    target_changed = True
    target_person_id = None
    target_lost_track_id = True

    # initialize deep sort
    encoder = gdet.create_box_encoder(os.getenv('HOME') + '/wego_ws/src/scout_ros/scout_bringup/model_data/mars-small128.pb', batch_size=1)

    # calculate cosine distance metric and initialize tracker
    metric_person = nn_matching.NearestNeighborDistanceMetric("cosine", 0.4, None)
    tracker_person = Tracker(metric_person)

    # load configuration for object detector
    config = ConfigProto()
    config.gpu_options.allow_growth = True

    saved_model_loaded_person = tf.saved_model.load(FLAGS.weights_person, tags=[tag_constants.SERVING])
    infer_person = saved_model_loaded_person.signatures['serving_default']

    saved_model_loaded_marker = tf.saved_model.load(FLAGS.weights_marker, tags=[tag_constants.SERVING])
    infer_marker = saved_model_loaded_marker.signatures['serving_default']

    # # Depth camera class 불러오기
    dc = DepthCamera()
    
    ##### marker 종류 선택 #####
    print("====================================================================")
    print(f"[Test Case] marker 1: '{FLAGS.marker1}', marker 2:'{FLAGS.marker2}' ---> Start")
    
    switching_time = time.time()
    marker_dic = {0: FLAGS.marker1, 1: FLAGS.marker2}
    marker_idx = 0
    frame_num = 0
    
    # while video is running
    while not rospy.is_shutdown():

        # depth camera 사용
        return_value, depth_frame, frame = dc.get_frame()

        if return_value:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame)
        else:
            print('Video has ended or failed, try a different video format!')
            break
        
        # 프레임 넘버 1 증가
        frame_num +=1
        print('Frame #: ', frame_num)
        target_marker = marker_dic[marker_idx]
        
        # 프레임 이미지 정보
        image_data = cv2.resize(frame, (FLAGS.size, FLAGS.size))
        image_data = image_data / 255.
        image_data = image_data[np.newaxis, ...].astype(np.float32)
        
        batch_data = tf.constant(image_data)
        
        # 다시 track id 설정
        target_marker_bboxes = []
        if target_changed or target_lost_track_id:
            detections_marker = detect_objects(infer_marker, batch_data, frame, encoder, idx_to_name_marker)
            # draw bbox for all detections and find target marker's bbox
            target_marker_bboxes = [detection.to_tlbr() for detection in detections_marker if detection.get_class() == target_marker]
            for bbox in target_marker_bboxes:
                color = marker_colors[name_to_idx_marker[target_marker]]
                cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
                cv2.rectangle(frame, (int(bbox[0]), int(bbox[1]-30)), (int(bbox[0])+34, int(bbox[1])), color, -1)
                cv2.putText(frame, target_marker, (int(bbox[0]), int(bbox[1]-10)), 0, 0.75, (255,255,255), 2)

        # target marker를 가진 사람 detection 후 tracking
        detections_person = detect_objects(infer_person, batch_data, frame, encoder, idx_to_name_person)
        
        # Call the tracker
        tracker_person.predict()
        tracker_person.update(detections_person, frame_num)

        # tracker.lost가 True라면 Target lost
        if tracker_person.lost:
            # target_changed = True
            target_person_id = None
            target_lost_track_id = True
            print('There are no objects to track.')
        
        # 추적 알고리즘
        lost = True # 트래킹할 객체 중 타깃이 없다면 True, 아니면 False

        # 추적할 객체가 있다면 동작
        for track in tracker_person.tracks:
            if not track.is_confirmed() or track.time_since_update > 1: continue
            
            bbox = track.to_tlbr()
            track_id = track.track_id
            
            # target marker가 변한 경우, track.id 설정 후 그 사람 tracking
            # 아니라면, 기존의 track.id를 가진 사람 tracking
            if target_changed or target_lost_track_id:
                for marker_bbox in target_marker_bboxes:
                    if marker_bbox[0] >= bbox[0] and marker_bbox[1] >= bbox[1] and marker_bbox[2] <= bbox[2] and marker_bbox[3] <= bbox[3]:
                        print(f"[Result] Target marker: '{target_marker}', Delay: {time.time() - switching_time} sec")
                        target_person_id = track_id
                        target_changed = False
                        target_lost_track_id = False
                        break

            if target_changed: continue # target marker 바뀌고 다시 track id 설정 안된 경우, track 하지 않기
            if track_id != target_person_id: continue  # target id에 해당하지 않은 사람 객체 무시
            
            # target id에 해당하는 사람 객체 tracking            
            lost = False
            
            # draw target person's bbox on screen
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), person_color, 2)
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1]-30)), (int(bbox[0])+85, int(bbox[1])), person_color, -1)
            cv2.putText(frame, "Person-" + str(track_id), (int(bbox[0]), int(bbox[1]-10)), 0, 0.75, (255,255,255), 2)

            break # 타깃 사람 찾으면 break
        
        if lost: 
            target_person_id = None
            target_lost_track_id = True
            print('There are no objects to track.')

        result = np.asarray(frame)
        result = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        # out.write(result) # RECORD VIDEO
        cv2.imshow("Output Video", result)
        if frame_num == 330:
            cv2.destroyAllWindow()
            print(f"frame #: {frame_num} ---> End")
        keyboard = cv2.waitKey(1) & 0xFF
        if keyboard == 27 or keyboard == 113: # ESC 또는 q 입력으로 프로그램 종료
            dc.release()
            # out.release() # RECORD VIDEO
            cv2.destroyAllWindows()
            print(f"key 'ESC' or 'q' ---> End")
            break
        
        # elif 48 <= keyboard <= 52: # 0, 1, 2, 3, 4 키보드 입력으로 타겟 마커 변경
        if frame_num % 30 == 0:
            marker_idx = 1 - marker_idx
            print(f"[frame #: {frame_num}] Previous Marker: '{target_marker}' ---> Target Marker: '{marker_dic[marker_idx]}'")
            target_marker = marker_dic[marker_idx]
            target_changed = True
            switching_time = time.time()
    
    print(f"[Test Case] marker 1 '{FLAGS.marker1}', marker 2 '{FLAGS.marker2}' ---> End")
    print("====================================================================")
        
        
if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass