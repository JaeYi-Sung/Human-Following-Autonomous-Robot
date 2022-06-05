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

# math
import math

# comment out below line to enable tensorflow logging outputs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import time
import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
from absl import app, flags, logging
from absl.flags import FLAGS
import core.utils as utils
from core.yolov4 import filter_boxes
from tensorflow.python.saved_model import tag_constants
from core.config import cfg
from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
# deep sort imports
from deep_sort import preprocessing, nn_matching
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet
from target import Target
import predict_object

# 웹캠을 사용하려면 True, D435를 사용하려면 False
use_webcam = False

flags.DEFINE_string('framework', 'tf', '(tf, tflite, trt)')
flags.DEFINE_string('weights_person', os.getenv('HOME') + '/wego_ws/src/scout_ros/scout_bringup/checkpoints/yolov4-tiny-416-person',
                    'path to weights file')
flags.DEFINE_string('weights_marker', os.getenv('HOME') + '/wego_ws/src/scout_ros/scout_bringup/checkpoints/yolov4-tiny-416-marker',
                    'path to weights file')
flags.DEFINE_string('weights_interaction', os.getenv('HOME') + '/wego_ws/src/scout_ros/scout_bringup/checkpoints/yolov4-tiny-416-interaction',
                    'path to weights file')
flags.DEFINE_integer('size', 416, 'resize images to')
flags.DEFINE_boolean('tiny', True, 'yolo or yolo-tiny')
flags.DEFINE_string('model', 'yolov4', 'yolov3 or yolov4')
flags.DEFINE_string('output', None, 'path to output video')
flags.DEFINE_string('output_format', 'XVID', 'codec used in VideoWriter when saving video to file')
flags.DEFINE_boolean('dont_show', False, 'dont show video output')
flags.DEFINE_boolean('info', False, 'show detailed info of tracked objects')
flags.DEFINE_string('cfg_yolo_classes_person', './scout_bringup/data/classes/person.names', 'path to cfg yolo classes file (person.names)')
flags.DEFINE_string('cfg_yolo_classes_marker', './scout_bringup/data/classes/marker.names', 'path to cfg yolo classes file (marker.names)')
flags.DEFINE_string('cfg_yolo_classes_interaction', './scout_bringup/data/classes/interaction.names', 'path to cfg yolo classes file (interaction.names)')

def main(_argv):
    
    cmap = plt.get_cmap('tab20b')
    colors = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]
    #start_time = time.time()

    # window에 포커스되어야 waitKey 적용됨
    WindowName = "Output Video"
    view_window = cv2.namedWindow(WindowName)
    # view_window = cv2.namedWindow(WindowName,cv2.WINDOW_NORMAL)
    # These two lines will force the window to be on top with focus.
    # cv2.setWindowProperty(WindowName,cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)
    # cv2.setWindowProperty(WindowName,cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_NORMAL)
    
    # y 증가: 아래 x 증가: 오른쪽
    # draw bbox on screen
    def draw_bbox(bbox, frame, class_name, *track_id):
        if len(track_id) != 0: # track_id 가 있으면
            track_id = track_id[0]
            color = colors[int(track_id) % len(colors)]
            color = [i * 255 for i in color]
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1]-30)), (int(bbox[0])+(len(class_name)+len(str(track_id)))*17, int(bbox[1])), color, -1)
            cv2.putText(frame, class_name + "-" + str(track_id),(int(bbox[0]), int(bbox[1]-10)),0, 0.75, (255,255,255),2)
        else: # track_id 가 없으면, 클래스 이름
            tmp = int(class_name) if class_name.isdigit() else (len(class_name) + 10)
            color = colors[tmp % len(colors)]
            color = [i * 255 for i in color]
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1]-30)), (int(bbox[0])+(len(class_name))*17, int(bbox[1])), color, -1)
            cv2.putText(frame, class_name,(int(bbox[0]), int(bbox[1]-10)),0, 0.75, (255,255,255), 2)

    # draw bbox for all detections and find target marker's bbox
    def find_target_marker_bboxes(detections, target):
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

    # target marker가 변한 경우, track.id 설정 후 그 사람 tracking
    # 아니라면, 기존의 track.id를 가진 사람 tracking
    def track_person(tracker, detections, target, target_marker_bboxes, interaction_closer, closer_dist):
        nonlocal x, y, z, th, speed, turn, frame_num, key, cx, cy, final_closer # go

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
            key = 'stop'
            target.lost_track_id = True

            print('There are no objects to track.')
            return
        
        lost = True

        # 추적할 객체가 있다면 동작
        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            
            bbox = track.to_tlbr()
            class_name = track.get_class()
            
            # target marker가 변한 경우, target id를 변경 (find the person who has a targeted marker)
            if target.changed or target.lost_track_id:
                for marker_bbox in target_marker_bboxes:
                    if marker_bbox[0] >= bbox[0] and marker_bbox[1] >= bbox[1] and marker_bbox[2] <= bbox[2] and marker_bbox[3] <= bbox[3]:
                        target.set_track_id(track.track_id)
                        print("target id: ", target.track_id)
                        break
                    
            # target id에 해당하지 않은 사람 객체 무시
            if track.track_id != target.track_id: continue
            
            lost = False
            # target id에 해당하는 사람 객체 tracking
            # cx, cy 계산 추가
            w, h = int(bbox[2]-bbox[0]), int(bbox[3]-bbox[1])
            cx, cy = int(w/2 + bbox[0]), int(h/2 + bbox[1])
            
            # 사람과 로봇의 거리: person_distance
            if not use_webcam:
                person_distance = person_dist(depth_frame, cx, cy, h)
                print('person distance : ', person_dist(depth_frame, cx, cy, h))

            if(final_closer) :
                print('interaction closer')
                key,speed,turn = drive4(cx, left_limit, right_limit, turn, frame, speed, max_speed, min_speed, max_turn, closer_dist, stable_max_dist, person_distance)

            else : 
                if person_distance < stable_min_dist: # 로봇과 사람의 거리가 직진 안전 구간 최솟값보다 작을 때 정지
                    print('Too Close')
                    key = 'stop'
                else:
                    print('key is NOT None')                 
                    key,speed,turn = drive4(cx, left_limit, right_limit, turn, frame, speed, max_speed, min_speed, max_turn, stable_min_dist, stable_max_dist, person_distance)
            
            # draw bbox on screen
            draw_bbox(bbox, frame, class_name, track.track_id)
        
        if lost: target.lost_track_id = True

    def is_interaction(detections, interaction_closer, interaction_stop, start_time_c, start_time_s):
        nonlocal cx, cy

        person_cx, person_cy = cx, cy
        check_closer = False
        check_stop = False

        print(detections)

        for detection in detections:
            
            gs_bbox = detection.to_tlbr()
            class_name = detection.get_class()

            # 사람 bbox와 손bbox 사이의 거리
            gs_w, gs_h = int(gs_bbox[2] - gs_bbox[0]), int(gs_bbox[3] - gs_bbox[1])
            gs_cx, gs_cy = int(gs_w/2 + gs_bbox[0]), int(gs_h/2 + gs_bbox[1])
            person_gs_dist = math.sqrt(math.pow(person_cx - gs_cx, 2) + math.pow(person_cy - gs_cy, 2))

            print("is_interaction FUNCTION: person_gs_dist - ", person_gs_dist)

            if (100 <= person_gs_dist <= 500) :
                draw_bbox(gs_bbox, frame, class_name)
                print(f"is_interaction FUNCTION: interaction_\"{class_name}\" TRUE")
                if (class_name == 'closer'):
                    check_closer = True
                    if (interaction_closer == False) :
                        start_time_c = time.time()
                        interaction_closer = True

                if (class_name == 'stop'):
                    check_stop = True
                    if (interaction_stop == False) :
                        start_time_s = time.time()
                        interaction_stop = True

        if(not check_closer) :
            print(f"is_interaction FUNCTION: interaction_closer FALSE")
            interaction_closer = False
            final_closer = False
            start_time_c = -1

        if(not check_stop) :
            print(f"is_interaction FUNCTION: interaction_stop FALSE")
            interaction_stop = False
            start_time_s = -1

        return interaction_closer, interaction_stop, start_time_c, start_time_s


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

    saved_model_loaded_interaction = tf.saved_model.load(FLAGS.weights_interaction, tags=[tag_constants.SERVING])
    infer_interaction = saved_model_loaded_interaction.signatures['serving_default']

    
    # 로봇 모터 제어를 위한 초깃값 설정
    x = 0
    y = 0
    z = 0
    th = 0
    speed = 0.1
    turn = 1

    # 로봇의 최대,최소 속도 설정
    # <!--선속도--!>
    max_speed = 0.4
    min_speed = 0.2

    # <!--각속도--!>
    max_turn = 0.2
    min_turn = 0.1

    # 변수 추가
    cx, cy, h = 0, 0, 0
    frame_num = 0
    key =''

    # 직진 안전 구간 최대/최소값
    stable_max_dist = 2500
    stable_min_dist = 2000
    closer_dist = 1000

    # Depth camera class 불러오기
    if not use_webcam:
        dc = DepthCamera()

    # 장애물 영역 기본값 받아오기
    default = Default_dist()

    # ROS class init
    go = scout_pub_basic()
    rate = rospy.Rate(60)
    
    # 타겟 설정을 위한 객체
    target = Target("0")

    # interaction 동작 인식을 위한 파라미터
    start_time_c = -1
    start_time_s = -1
    limit_time = 2
    interaction_stop = False
    interaction_closer = False
    final_closer = False

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
        
                # 좌/우 회전 한곗값 설정
        left_limit = frame.shape[1]//2 - 70
        right_limit = frame.shape[1]//2 + 70
        
        # 프레임 넘버 1 증가
        frame_num +=1
        print('Frame #: ', frame_num)
        
        # 장애물 회피를 위한 ROI 디폴트 세팅하기 (현재는 10프레임만) 추가
        if frame_num < 11 :
            default.default_update(depth_frame)
            continue
        
        # fps 측정을 위한 시작 시간 측정
        frame_start_time = time.time()
        
        # 프레임 이미지 정보
        image_data = cv2.resize(frame, (input_size, input_size))
        image_data = image_data / 255.
        image_data = image_data[np.newaxis, ...].astype(np.float32)
        
        batch_data = tf.constant(image_data)
        
        target_marker_bboxes = []
        
        # target marker 가 변한 경우
        if target.changed or target.lost_track_id:
            detections_marker = predict_object.detection(infer_marker, batch_data, frame, encoder, FLAGS.cfg_yolo_classes_marker)
            target_marker_bboxes = find_target_marker_bboxes(detections_marker, target)

        # target marker를 가진 사람 detection 후 tracking
        # person detection
        detections_person = predict_object.detection(infer_person, batch_data, frame, encoder, FLAGS.cfg_yolo_classes_person)
        
        # target marker가 변한 경우, track.id 설정 후 그 사람 tracking
        # 아니라면, 기존의 track.id를 가진 사람 tracking
        track_person(tracker_person, detections_person, target, target_marker_bboxes, final_closer, closer_dist) # 사람 따라가기

        # Interaction
        detections_interaction = predict_object.detection(infer_interaction, batch_data, frame, encoder, FLAGS.cfg_yolo_classes_interaction)
        

        interaction_closer, interaction_stop, start_time_c, start_time_s = is_interaction(detections_interaction, interaction_closer, interaction_stop, start_time_c, start_time_s)

        if(start_time_c != -1) :
            now_time = time.time()
            gs_time = now_time - start_time_c
            print(gs_time)

            if (gs_time > limit_time) :
                if(interaction_closer) :
                    print(f"MAIN FUNCTION: final_closer TRUE")
                    final_closer = True
                    #interaction_closer = False
        
        elif(start_time_s != -1) :
            now_time = time.time()
            gs_time = now_time - start_time_s
            print(gs_time)

            if (gs_time > limit_time) :
                if(interaction_stop) :
                    print(f"MAIN FUNCTION: key STOP")
                    #interaction_stop = False
                    key = 'stop'
                # start_time = -1


        # 주행 알고리즘(drive)를 거치고 나온 속도/방향을 로봇에 전달
        x,y,z,th,speed,turn = key_move(key,x,y,z,th,speed,turn)

        print('key: ', key)
        print('x: {}, y: {}, th: {}, speed: {}, turn: {}'.format(x,y,th,speed,turn))
        
        # 화면 중심 표시
        cv2.circle(frame, (320, 240), 10, (255, 255, 255))

        # 좌우 회전 구분선 그리기
        cv2.line(frame, (left_limit,0), (left_limit,frame.shape[0]), (255,0,0))
        cv2.line(frame, (right_limit,0), (right_limit,frame.shape[0]), (255,0,0))

        # ROS Rate sleep
        rate.sleep()

        '''
        box_center_roi = np.array((depth_frame[cy-10:cy+10, cx-10:cx+10]),dtype=np.float64)
        cv2.rectangle(frame, (cx-10, cy+10), (cx+10, cy-10), (255, 255, 255), 2)
        '''

        safe_roi = np.array([[400, 400], [240, 400], [160, 480], [480, 480]])
        #safe_roi = np.array([[240, 420], [400, 420], [480, 160], [480, 480]])
        cv2.polylines(frame, [safe_roi], True, (255, 255, 255), 2)
        cv2.rectangle(frame, (205, 445), (195, 435), (255, 0, 0), 5)
        cv2.rectangle(frame, (245, 405), (235, 395), (255, 0, 0), 5)
        cv2.rectangle(frame, (405, 405), (395, 395), (255, 0, 0), 5)
        cv2.rectangle(frame, (445, 445), (435, 435), (255, 0, 0), 5)

        # calculate frames per second of running detections
        fps = 1.0 / (time.time() - frame_start_time)
        print("FPS: %.2f" % fps)
        
        result = np.asarray(frame)
        result = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        if not FLAGS.dont_show:
            cv2.imshow(WindowName, result)
         
        keyboard = cv2.waitKey(1) & 0xFF

        #  0, 1, 2, 3, 4 입력으로 타겟 마커 변경
        if 48 <= keyboard <= 52:
            target_marker = str((keyboard - 48) * 10)
            print(f"key \"{chr(keyboard)}\" 입력 ---> 마커 \"{target_marker}\" 선택")
            target.set_target(target_marker)
        
        # ESC 또는 q 입력으로 프로그램 종료
        if keyboard == 27 or keyboard == 113:
            dc.release()
            cv2.destroyAllWindows()
            print(f"key 'ESC or q' 입력 ---> 끝내기")
            break

if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass