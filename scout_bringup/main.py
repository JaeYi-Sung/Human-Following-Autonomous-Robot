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
import mediapipe as mp
from target import Target

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

# bounding box 색깔 지정
cmap = plt.get_cmap('tab20b')
colors = [cmap(i)[:3] for i in np.linspace(0, 1, 6)]
marker_colors = [[i * 255 for i in color] for color in colors[:5]]
person_color = [i * 255 for i in colors[5]]

# detection indexing
idx_to_name_marker = {0: "0", 1: "10", 2: "20", 3: "30", 4: "40"}
idx_to_name_person = {0: "person"}

# 시간 제한 상수 설정
TIME = 1 # 손동작 1초간 유지시 의미 있는 동작으로 인식
REST = 3 # 손동작 인식 후 3초간 인식하지 않기
CLOSER = 10 # 10초 동안 더 가까운 거리에서 주행

# interaction을 위한 상수 설정 및 
compare_index = [[7, 8], [11, 12], [15, 16], [19, 20]]  # 검지, 중지, 약지, 새끼 landmarks
gestures = ["fist", "pointing", "usual"]
gesture_time = [0, 0, 0, 0] # fist_time, pointing_time, inactive_time, closer_time = 0, 0, 0, 0

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


def dist(x1, y1, x2, y2):
    return math.sqrt(math.pow(x1 - x2, 2) + math.pow(y1 - y2, 2))

def check_hand_gesture(hand_landmarks, frame):
    
    finger_dist = 0b0000
    wrist = hand_landmarks.landmark[0]  # 손목
    wrist_x, wrist_y = wrist.x, wrist.y  # 손목 x, y 좌표
    h, w, _ = frame.shape

    for i in range(4):
        finger_dip = hand_landmarks.landmark[compare_index[i][0]]  # 손가락 첫번째 마디
        finger_tip = hand_landmarks.landmark[compare_index[i][1]]  # 손가락 끝

        # 손가락 거리
        if dist(wrist_x, wrist_y, finger_dip.x, finger_dip.y) < dist(wrist_x, wrist_y, finger_tip.x, finger_tip.y):
            finger_dist = finger_dist | (1 << i)
    
    if finger_dist == 0 or finger_dist == 1:
        cv2.putText(frame, gestures[finger_dist], (round(wrist_x * w), round(wrist_y * h)), cv2.FONT_HERSHEY_PLAIN, 4, (255, 215, 0), 4)        
        return finger_dist
    return -1

def main(_argv):
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
    mode = 0 # 정지 상태 (1: 주행 상태, 0: 정지 상태)
    closer = False
    # fist_time, pointing_time, inactive_time, closer_time = 0, 0, 0, 0 # fist 인식 시작 시간, pointing 인식 시작 시간, 휴면 인식 시작 시간

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

    # Depth camera class 불러오기
    dc = DepthCamera()

    # 장애물 영역 기본값 받아오기
    default = Default_dist()

    # ROS class init
    go = scout_pub_basic()
    rate = rospy.Rate(60)
    
    # 타겟 설정을 위한 객체
    target = Target("0")
    
    # mediapipe
    mp_drawing = mp.solutions.drawing_utils
    mp_holistic = mp.solutions.holistic
    holistic = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)

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
        image_data = cv2.resize(frame, (FLAGS.size, FLAGS.size))
        image_data = image_data / 255.
        image_data = image_data[np.newaxis, ...].astype(np.float32)
        
        batch_data = tf.constant(image_data)
        
        # 다시 track id 설정
        target_marker_bboxes = []
        if target.changed or target.lost_track_id:
            detections_marker = detect_objects(infer_marker, batch_data, frame, encoder, idx_to_name_marker)
            # draw bbox for all detections and find target marker's bbox
            target_class = target.marker
            target_marker_bboxes = [detection.to_tlbr() for detection in detections_marker if detection.get_class() == target_class]
            for bbox in target_marker_bboxes:
                color = marker_colors[int(target_class) // 10]
                cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
                cv2.rectangle(frame, (int(bbox[0]), int(bbox[1]-30)), (int(bbox[0])+34, int(bbox[1])), color, -1)
                cv2.putText(frame, target_class, (int(bbox[0]), int(bbox[1]-10)), 0, 0.75, (255,255,255), 2)

        # target marker를 가진 사람 detection 후 tracking
        detections_person = detect_objects(infer_person, batch_data, frame, encoder, idx_to_name_person)
        
        # Call the tracker
        tracker_person.predict()
        tracker_person.update(detections_person, frame_num)

        # <st-mini 제어를 위한 Publisher code>
        go.update(x, y, z, th, speed, turn)
        
        # tracker.lost가 True라면 Target lost
        
        if tracker_person.lost:
            go.sendMsg(frame_num % 2)
            target.lost_track_id = True
            key = 'stop'
            print('There are no objects to track.')
            return
        
        # 추적 알고리즘
        go.sendMsg(1)
        lost = True # 트래킹할 객체 중 타깃이 없다면 True, 아니면 False

        # 추적할 객체가 있다면 동작
        for track in tracker_person.tracks:
            if not track.is_confirmed() or track.time_since_update > 1: continue
            
            bbox = track.to_tlbr()
            track_id = track.track_id
            
            # target marker가 변한 경우, track.id 설정 후 그 사람 tracking
            # 아니라면, 기존의 track.id를 가진 사람 tracking
            if target.changed or target.lost_track_id:
                for marker_bbox in target_marker_bboxes:
                    if marker_bbox[0] >= bbox[0] and marker_bbox[1] >= bbox[1] and marker_bbox[2] <= bbox[2] and marker_bbox[3] <= bbox[3]:
                        target.set_track_id(track_id)
                        break
            
            if target.changed: continue # target marker 바뀌고 다시 track id 설정 안된 경우, track 하지 않기
            if track_id != target.track_id: continue  # target id에 해당하지 않은 사람 객체 무시
            
            # target id에 해당하는 사람 객체 tracking            
            lost = False
            # # target person의 bbox 저장
            # target.track_bbox = bbox
            
            # target person의 bbox의 w, h, cx, cy 계산
            w, h = int(bbox[2]-bbox[0]), int(bbox[3]-bbox[1])
            cx, cy = int(w/2 + bbox[0]), int(h/2 + bbox[1])
            
            # 사람과 로봇의 거리: person_distance
            person_distance = person_dist(depth_frame, cx, cy, h)
            print('person distance : ', person_distance)

            # 주행 함수 호출 또는 주행 키 설정
            if mode: # mode가 1이라면 주행, 0 이라면 비주행
                if not closer: # closer 가 False일 때, 평소 거리로 주행
                    key, speed, turn = drive(cx, frame, turn, speed, person_distance)
                elif closer_time - now_time <= CLOSER: # closer가 True일 때, CLOSER 동안 더 가깝게 주행하는 함수 호출
                    key, speed, turn = drive(cx, frame, turn, speed, 1.1 * person_distance)
                else: # closer가 True이고, CLOSER 시간 초과 시, 초기화
                    closer_time = 0
                    closer = False
            else: key = 'stop'
            
            # draw target person's bbox on screen
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), person_color, 2)
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1]-30)), (int(bbox[0])+85, int(bbox[1])), person_color, -1)
            cv2.putText(frame, "Person-" + str(track_id), (int(bbox[0]), int(bbox[1]-10)), 0, 0.75, (255,255,255), 2)
            
            ##### Interaction ##### 
            if gesture_time[2] and now_time - gesture_time[2] >= REST: # 휴면 시간 초과했다면
                gesture_time[2] = 0 # 초기화
            else: # 휴면 시간 초과하지 않았고, 다른 동작이 인식되지 않았다면, 제스쳐 인식
                gesture_idx = -1 # usual로 초기화
                
                # 타깃 사용자의 몸 확인
                frame_h, frame_w, _ = frame.shape
                holistic_results = holistic.process(frame)

                if holistic_results.pose_landmarks:
                    # 타깃 사용자만의 몸 landmark 그리고, 손동작 인식
                    left_shoulder = holistic_results.pose_landmarks.landmark[11]
                    right_shoulder = holistic_results.pose_landmarks.landmark[12]
                    mid_shoulder = ((left_shoulder.x + right_shoulder.x) * frame_w / 2, (left_shoulder.y + right_shoulder.y) * frame_h / 2)
                    if (bbox[0] <= mid_shoulder[0] <= bbox[2]) and (bbox[1] <= mid_shoulder[1] <= bbox[3]):
                        mp_drawing.draw_landmarks(frame, holistic_results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
                        # 우선 순위: 왼손 < 오른손
                        if holistic_results.left_hand_landmarks:  # 왼손
                            mp_drawing.draw_landmarks(frame, holistic_results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
                            gesture_idx = check_hand_gesture(holistic_results.left_hand_landmarks, frame)
                        if holistic_results.right_hand_landmarks:  # 오른손
                            mp_drawing.draw_landmarks(frame, holistic_results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
                            gesture_idx = max(gesture_idx, check_hand_gesture(holistic_results.right_hand_landmarks, frame))
                            # 우선 순위: usual < fist < pointing
                if gesture_idx == -1: # usual인 경우
                    gesture_time[0], gesture_time[1] = 0, 0 # 모든 손동작 인식 시작 시간 초기화
                else: # 인식된 손동작 있는 경우
                    if gesture_time[gesture_idx] == 0: gesture_time[gesture_idx] = now_time # 처음 인식했다면, 인식 시작 시간 저장
                    elif now_time - gesture_time[gesture_idx] >= TIME: # 처음 설정이 아니고, 의미 있는 시간인 1초 동안 프레임에서 인식
                        mode[gesture_idx] = 1 - mode[gesture_idx] # 주행 상태 변경 toggle
                        if gesture_idx == 0: # fist인 경우
                            mode = 1 - mode
                        else: # pointing인 경우
                            closer = True
                            closer_time = now_time
                        gesture_time[gesture_idx] = 0
                        gesture_time[2] = 0
                    gesture_time[1 - gesture_idx] = 0 # 다른 손동작 초기화
                    
            break # 타깃 사람 찾으면 break
        
        if lost: 
            target.lost_track_id = True
            key = 'stop'
            print('There are no objects to track.')
        
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
        if keyboard == 27 or keyboard == 113: # ESC 또는 q 입력으로 프로그램 종료
            dc.release()
            cv2.destroyAllWindows()
            print(f"key 'ESC' or 'q' 입력 ---> 끝내기")
            break
        elif 48 <= keyboard <= 52: # 0, 1, 2, 3, 4 키보드 입력으로 타겟 마커 변경
            target_marker = str((keyboard - 48) * 10)
            print(f"key '{chr(keyboard)}' 입력 ---> 마커 '{target_marker}' 선택")
            target.set_target(target_marker)
        
if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass