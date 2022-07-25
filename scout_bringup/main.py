#! /usr/bin/env python

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

# comment out below line to enable tensorflow logging outputs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import time
import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
from absl import app, flags
from absl.flags import FLAGS
import core.utils as utils
from tensorflow.python.saved_model import tag_constants
from core.config import cfg
from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.compat.v1 import ConfigProto
# deep sort imports
from deep_sort import nn_matching
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet
from target import Target
import predict_object
import interaction

flags.DEFINE_string('weights_person', os.getenv('HOME') + '/wego_ws/src/scout_ros/scout_bringup/checkpoints/yolov4-tiny-416-person',
                    'path to weights file')
flags.DEFINE_string('weights_marker', os.getenv('HOME') + '/wego_ws/src/scout_ros/scout_bringup/checkpoints/yolov4-tiny-416-marker',
                    'path to weights file')
flags.DEFINE_integer('size', 416, 'resize images to')

# bounding box 색깔 지정
cmap = plt.get_cmap('tab20b')
colors = [cmap(i)[:3] for i in np.linspace(0, 1, 6)]
marker_colors = [i * 255 for i in colors[:5]]
person_color = [i * 255 for i in colors[5]]

# detection indexing
idx_to_name_marker = {0: "0", 1: "10", 2: "20", 3: "30", 4: "40"}
idx_to_name_person = {0: "person"}

# 로봇 모터 제어를 위한 초깃값 설정
x = 0
y = 0
z = 0
th = 0
speed = 0.1
turn = 1

# 변수 추가
frame_num = 0
key =''

# 주행 상태 설정 변수
mode = 0 # 주행 상태 (0: 주행 상태, 1: 정지 상태)
closer = False
fist_time, pointing_time, inactive_time, closer_time = 0, 0, 0, 0 # fist 인식 시작 시간, pointing 인식 시작 시간, 휴면 인식 시작 시간

# 시간 제한 상수 설정
TIME = 1 # 손동작 1초간 유지시 의미 있는 동작으로 인식
REST = 3 # 손동작 인식 후 3초간 인식하지 않기
CLOSER = 10 # 10초 동안 더 가까운 거리에서 주행

# y 증가: 아래 x 증가: 오른쪽
# draw bbox on screen
def draw_bbox(bbox, frame, class_name, *track_id):
    if len(track_id) != 0: # track_id 가 있으면
        track_id = track_id[0]
        color = person_color
        text = class_name + "-" + str(track_id)
    else: # track_id 가 없으면
        color = marker_colors[int(class_name) // 10]
        text = class_name
        
    cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
    cv2.rectangle(frame, (int(bbox[0]), int(bbox[1]-30)), (int(bbox[0])+(len(text))*17, int(bbox[1])), color, -1)
    cv2.putText(frame, text, (int(bbox[0]), int(bbox[1]-10)), 0, 0.75, (255,255,255), 2)

# target marker가 변한 경우, track.id 설정 후 그 사람 tracking
# 아니라면, 기존의 track.id를 가진 사람 tracking
def track_person(tracker, detections, target, go, target_marker_bboxes, frame, now_time):
    global x, y, z, th, speed, turn, frame_num, key, closer, closer_time

    # Call the tracker
    tracker.predict()
    tracker.update(detections, frame_num)

    # <st-mini 제어를 위한 Publisher code>
    go.update(x, y, z, th, speed, turn)
    
    # tracker.lost가 True라면 Target lost
    if tracker.lost:
        go.sendMsg(frame_num % 2)
    else :
        go.sendMsg(1)

    # 추적 알고리즘
    if tracker.lost: # 추적할 객체가 없다면 정지
        target.lost_track_id = True
        key = 'stop'
        print('There are no objects to track.')
        return
    
    lost = True
    
    # 추적할 객체가 있다면 동작
    for track in tracker.tracks:
        if not track.is_confirmed() or track.time_since_update > 1: continue
        
        bbox = track.to_tlbr()
        class_name = track.get_class()
        track_id = track.track_id
        
        # target marker가 변한 경우, target id를 변경 (find the person who has a targeted marker)
        if target.changed or target.lost_track_id:
            for marker_bbox in target_marker_bboxes:
                if marker_bbox[0] >= bbox[0] and marker_bbox[1] >= bbox[1] and marker_bbox[2] <= bbox[2] and marker_bbox[3] <= bbox[3]:
                    target.set_track_id(track_id)
                    # print("target id: ", target.track_id)
                    break
        
        if target.changed: continue # target marker 바뀌고 다시 track id 설정 안된 경우, track 하지 않기
        
        # target id에 해당하지 않은 사람 객체 무시
        if track_id != target.track_id: continue
        
        # target id에 해당하는 사람 객체 tracking            
        lost = False
        
        # target person의 bbox 저장
        target.track_bbox = bbox
        
        # cx, cy 계산 추가
        w, h = int(bbox[2]-bbox[0]), int(bbox[3]-bbox[1])
        cx, cy = int(w/2 + bbox[0]), int(h/2 + bbox[1])
        
        # 사람과 로봇의 거리: person_distance
        person_distance = person_dist(depth_frame, cx, cy, h)
        print('person distance : ', person_distance)
        
        # mode가 1이라면 주행, 0 이라면 비주행
        if mode:
            if not closer: # closer 가 False일 때, 평소 거리로 주행
                key, speed, turn = drive(cx, frame, turn, speed, person_distance)
            elif closer_time - now_time <= CLOSER: # closer가 True일 때, CLOSER 동안 더 가깝게 주행하는 함수 호출
                key, speed, turn = drive(cx, frame, turn, speed, 1.1 * person_distance)
            else: # closer가 True이고, CLOSER 시간 초과 시, 초기화
                closer_time = 0
                closer = False
        else: key = 'stop'
        
        # draw bbox on screen
        draw_bbox(bbox, frame, class_name, track_id)
    
    if lost: 
        target.lost_track_id = True
        key = 'stop'
        print('There are no objects to track.')

# draw bbox for all detections and find target marker's bbox
def find_target_marker_bboxes(detections, target, frame):
    marker_bbox_list = []

    for detection in detections:
        bbox = detection.to_tlbr()
        class_name = detection.get_class()
        
        # find target marker's bbox
        if class_name == target.marker: 
            marker_bbox_list.append(bbox)

        # draw bbox on screen
        draw_bbox(bbox, frame, class_name)
    
    return marker_bbox_list

def check_meaningful_gesture(target, frame, now_time, mode, closer, fist_time, pointing_time, inactive_time, closer_time):
    if now_time - inactive_time >= REST: # 휴면 시간 초과했다면
        inactive_time = 0 # 초기화
    else: # 휴면 시간 초과하지 않았고, 다른 동작이 인식되지 않았다면, 제스쳐 인식
        hand_gesture = interaction.recognize_hand_gesture(target, frame)

        # fist 인식
        if hand_gesture == "fist":
            if fist_time == 0: # 처음 fist 인식했다면
                fist_time = now_time # fist 인식 시작 시간 저장
            else: # 처음 설정이 아니고, 다음 1초 후 프레임에서도 인식
                if now_time - fist_time >= TIME: # 의미 있는 제스쳐
                    mode = 1 - mode # 로봇의 주행 상태 toggle
                    fist_time = 0
                    inactive_time = now_time # 휴면 시작 시간 저장
                    
            pointing_time = 0 # pointing 초기화
        
        # pointing 인식
        elif hand_gesture == "pointing":
            if pointing_time == 0:
                pointing_time = now_time
            else:
                if now_time - pointing_time >= TIME:
                    closer = True
                    pointing_time = 0
                    closer_time = now_time
                    inactive_time = now_time
                    
            fist_time = 0 # fist 인식 시작 시간 초기화
        
        # 어떤 손동작도 인식하지 못함 (normal) 
        else:
            fist_time, pointing_time = 0, 0 # 모든 손동작 인식 시작 시간 초기화
    return mode, closer, fist_time, pointing_time, inactive_time, closer_time

def main(_argv):
    global x, y, z, th, speed, turn, frame_num, key, mode, closer, fist_time, pointing_time, inactive_time, closer_time
    # Definition of the parameters
    max_cosine_distance = 0.4
    nn_budget = None

    # initialize deep sort
    model_filename = os.getenv('HOME') + '/wego_ws/src/scout_ros/scout_bringup/model_data/mars-small128.pb'
    encoder = gdet.create_box_encoder(model_filename, batch_size=1)

    # calculate cosine distance metric
    metric_person = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)

    # initialize tracker
    tracker_person = Tracker(metric_person)

    # load configuration for object detector
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    input_size = FLAGS.size

    saved_model_loaded_person = tf.saved_model.load(FLAGS.weights_person, tags=[tag_constants.SERVING])
    infer_person = saved_model_loaded_person.signatures['serving_default']

    saved_model_loaded_marker = tf.saved_model.load(FLAGS.weights_marker, tags=[tag_constants.SERVING])
    infer_marker = saved_model_loaded_marker.signatures['serving_default']

    # Depth camera class 불러오기

    dc = DepthCamera()

    # 장애물 영역 기본값 받아오기
    default = Default_dist()

    # ROS class init
    go = scout_pub_basic()
    rate = rospy.Rate(60)
    
    # 타겟 설정을 위한 객체
    target = Target("0")

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
        
        # 장애물 회피를 위한 ROI 디폴트 세팅하기 (현재는 10프레임만) 추가
        if frame_num < 11 :
            default.default_update(depth_frame)
            continue
        
        # 프레임 시작 시간 측정
        now_time = time.time()
        
        # 프레임 이미지 정보
        image_data = cv2.resize(frame, (input_size, input_size))
        image_data = image_data / 255.
        image_data = image_data[np.newaxis, ...].astype(np.float32)
        
        batch_data = tf.constant(image_data)
        
        target_marker_bboxes = []
        
        # 다시 track id 설정
        if target.changed or target.lost_track_id:
            detections_marker = predict_object.detection(infer_marker, batch_data, frame, encoder, idx_to_name_marker)
            target_marker_bboxes = find_target_marker_bboxes(detections_marker, target, frame)

        # target marker를 가진 사람 detection 후 tracking
        detections_person = predict_object.detection(infer_person, batch_data, frame, encoder, idx_to_name_person)

        # target marker가 변한 경우, track.id 설정 후 그 사람 tracking
        # 아니라면, 기존의 track.id를 가진 사람 tracking
        track_person(tracker_person, detections_person, target, go, target_marker_bboxes, frame, now_time) # 사람 따라가기
        
        # Interaction
        if not target.lost_track_id:
            mode, closer, fist_time, pointing_time, inactive_time, closer_time = check_meaningful_gesture(target, frame, now_time, mode, closer, fist_time, pointing_time, inactive_time, closer_time)
        
        # 주행 알고리즘(drive)를 거치고 나온 속도/방향을 로봇에 전달
        x, y, z, th, speed, turn = key_move(key, x, y, z, th, speed, turn)

        print('key: ', key)
        print('x: {}, y: {}, th: {}, speed: {}, turn: {}'.format(x, y, th, speed, turn))
        
        # ROS Rate sleep
        rate.sleep()
        
        # calculate frames per second of running detections
        fps = 1.0 / (time.time() - now_time)
        print("FPS: %.2f" % fps)
        
        result = np.asarray(frame)
        result = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.imshow("Output Video", result)
                    
        keyboard = cv2.waitKey(1) & 0xFF
        
        # ESC 또는 q 입력으로 프로그램 종료
        if keyboard == 27 or keyboard == 113:
            dc.release()
            cv2.destroyAllWindows()
            print(f"key 'ESC' or 'q' 입력 ---> 끝내기")
            break
        
        #  0, 1, 2, 3, 4 입력으로 타겟 마커 변경
        elif 48 <= keyboard <= 52:
            target_marker = str((keyboard - 48) * 10)
            print(f"key '{chr(keyboard)}' 입력 ---> 마커 '{target_marker}' 선택")
            target.set_target(target_marker)
        
if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass