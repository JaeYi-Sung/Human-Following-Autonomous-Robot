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
import predict_object

# 웹캠을 사용하려면 True, D435를 사용하려면 False
use_webcam = False

flags.DEFINE_string('framework', 'tf', '(tf, tflite, trt')
flags.DEFINE_string('weights_person', os.getenv('HOME') + '/wego_ws/src/scout_ros/scout_bringup/checkpoints/yolov4-tiny-416-person',
                    'path to weights file')
flags.DEFINE_string('weights_marker', os.getenv('HOME') + '/wego_ws/src/scout_ros/scout_bringup/checkpoints/yolov4-tiny-416-marker',
                    'path to weights file')
flags.DEFINE_integer('size', 416, 'resize images to')
flags.DEFINE_boolean('tiny', True, 'yolo or yolo-tiny')
flags.DEFINE_string('model', 'yolov4', 'yolov3 or yolov4')
flags.DEFINE_string('output', None, 'path to output video')
flags.DEFINE_string('output_format', 'XVID', 'codec used in VideoWriter when saving video to file')
flags.DEFINE_boolean('dont_show', False, 'dont show video output')
flags.DEFINE_boolean('info', False, 'show detailed info of tracked objects')
flags.DEFINE_string('cfg_yolo_classes_person', './scout_bringup/data/classes/person.names', 'path to cfg yolo classes file (person.names or marker.names)')
flags.DEFINE_string('cfg_yolo_classes_marker', './scout_bringup/data/classes/marker.names', 'path to cfg yolo classes file (person.names or marker.names)')

def main(_argv):
    
    cmap = plt.get_cmap('tab20b')
    colors = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]
    target_id = False

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
        else: # track_id 가 없으면
            color = colors[int(class_name) % len(colors)] # 클래스 이름
            color = [i * 255 for i in color]
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1]-30)), (int(bbox[0])+(len(class_name))*17, int(bbox[1])), color, -1)
            cv2.putText(frame, class_name,(int(bbox[0]), int(bbox[1]-10)),0, 0.75, (255,255,255),2)

    # draw bbox for all detections and find target marker's bbox
    def find_target_marker_bbox(detections, target_marker):
        target_marker_bbox_list = []        

        for detection in detections:
            bbox = detection.to_tlbr()
            print("find_target_marker - bbox:", bbox)
            class_name = detection.get_class()
            print("find_target_marker - class_name:", class_name)
            
            # find target marker's bbox
            if class_name == target_marker: 
                target_marker_bbox_list.append(bbox)

            # draw bbox on screen
            draw_bbox(bbox, frame, class_name)
        
        return target_marker_bbox_list

    # track people
    def track_person(tracker, detections, target_marker_bbox_list):
        nonlocal x, y, z, th, speed, turn, frame_num, key, target_id # go

        # Find tracking a target        
        track_id_list = []
        person_id_to_bbox = {}

        # Call the tracker
        tracker.predict()
        tracker.update(detections, frame_num)

        # <st-mini 제어를 위한 Publisher code>
        go.update(x, y, z, th, speed, turn)
        
        # tracker.lost가 True라면 Target lost
        if tracker.lost :
            go.sendMsg(frame_num%2)
        else :
            go.sendMsg(1)

        # 추적 알고리즘
        if tracker.lost: # 추적할 객체가 없다면 정지
            key = 'stop'
            print('There are no objects to track.')
        else: # 추적할 객체가 있다면 동작
            for track in tracker.tracks:
                if not track.is_confirmed() or track.time_since_update > 1:
                    continue 
                
                bbox = track.to_tlbr()
                class_name = track.get_class()
     
                if target_id == False:           
                    # Find person who has a targeted marker
                    for marker_bbox in target_marker_bbox_list:
                        if marker_bbox[0] >= bbox[0] and marker_bbox[1] >= bbox[1] and marker_bbox[2] <= bbox[2] and marker_bbox[3] <= bbox[3]:
                            target_id = track.track_id
                            print("iffffffffffff")
                            print("track.track_id ", track.track_id)
                            break
                        else:
                            print("elseeeeeeeeeeee")
                            target_id = False

                print('target_id: ',target_id)

                # cx, cy 계산 추가
                w, h = int(bbox[2]-bbox[0]), int(bbox[3]-bbox[1])
                cx, cy = int(w/2 + bbox[0]), int(h/2 + bbox[1])
                
                # 사람과 로봇의 거리: person_distance
                if not use_webcam:
                    person_distance = person_dist(depth_frame, cx, cy, h)
                    print('person distance : ', person_dist(depth_frame, cx, cy, h))

                # 직진 안전 구간 최대/최소값
                stable_max_dist = 2500
                stable_min_dist = 2000
                
                if person_distance < stable_min_dist: # 로봇과 사람의 거리가 직진 안전 구간 최솟값보다 작을 때 정지
                    print('Too Close')
                    key = 'stop'
                else:
                    print('key is NOT None')                 
                    key,speed,turn = drive4(cx, left_limit, right_limit, turn, frame, speed, max_speed, min_speed, max_turn, stable_min_dist, stable_max_dist, person_distance)
                
                # draw bbox on screen
                draw_bbox(bbox, frame, class_name, track.track_id)


    # Definition of the parameters
    max_cosine_distance = 0.4
    nn_budget = None

    # initialize deep sort
    model_filename = os.getenv('HOME') + '/wego_ws/src/scout_ros/scout_bringup/model_data/mars-small128.pb'
    encoder = gdet.create_box_encoder(model_filename, batch_size=1)

    # calculate cosine distance metric
    metric_person = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    # metric_marker = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)

    # initialize tracker
    tracker_person = Tracker(metric_person)
    # tracker_marker = Tracker(metric_marker)

    # load configuration for object detector
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    #session = InteractiveSession(config=config)
    #STRIDES, ANCHORS, NUM_CLASS, XYSCALE = utils.load_config(FLAGS)
    input_size = FLAGS.size


    saved_model_loaded_person = tf.saved_model.load(FLAGS.weights_person, tags=[tag_constants.SERVING])
    infer_person = saved_model_loaded_person.signatures['serving_default']

    saved_model_loaded_marker = tf.saved_model.load(FLAGS.weights_marker, tags=[tag_constants.SERVING])
    infer_marker = saved_model_loaded_marker.signatures['serving_default']

    
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
    # Depth camera class 불러오기
    if not use_webcam:
        dc = DepthCamera()

    # 장애물 영역 기본값 받아오기
    default = Default_dist()

    # ROS class init
    go = scout_pub_basic()
    rate = rospy.Rate(60)    

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
        frame_num +=1
        
        # 좌/우 회전 한곗값 설정
        left_limit = frame.shape[1]//2 - 70
        right_limit = frame.shape[1]//2 + 70

        if not use_webcam:
            # 장애물 회피를 위한 ROI 디폴트 세팅하기 (현재는 10프레임만) 추가
            if frame_num < 11 :
                default.default_update(depth_frame)
                continue
        
        print('Frame #: ', frame_num)
        #frame_size = frame.shape[:2]
        image_data = cv2.resize(frame, (input_size, input_size))
        image_data = image_data / 255.
        image_data = image_data[np.newaxis, ...].astype(np.float32)
        start_time = time.time()

        batch_data = tf.constant(image_data)
    
        # custom allowed classes (uncomment line below to customize tracker for only people)
        # allowed_classes_person = ['person']
        # allowed_classes_marker = ['0', '10', '20', '30', '40']


        detections_marker = predict_object.detection(infer_marker, batch_data, frame, encoder, FLAGS.cfg_yolo_classes_marker)

        """
        데모 코드에서 타겟 마커 설정
        frame_num 값에 따라 recognition_object 변경하도록 하기 => fps 변수로 두고 일정한 시간 간격으로 변경하도록 하기
        
        """
        target_marker = '0' # 타깃 마커 번호 '0' 으로 임시 설정 => 데모 코드에서 설정해주도록 해야 함

        """
        detection 한 것 marker에서 person 특정
        
        """

        # target_marker 값 변하면 해당 bbox 찾기
        target_marker_bbox_list = find_target_marker_bbox(detections_marker, target_marker) 
        
        detections_person = predict_object.detection(infer_person, batch_data, frame, encoder, FLAGS.cfg_yolo_classes_person)
        
        """
        target marker를 가진 사람 detection 후 tracking
        """
        track_person(tracker_person, detections_person, target_marker_bbox_list) # 사람 따라가기
        print("main target_id ", target_id)
        
        # 주행 알고리즘(drive)를 거치고 나온 속도/방향을 로봇에 전달
        x,y,z,th,speed,turn = key_move(key,x,y,z,th,speed,turn)

        print('key: ', key)
        print('key_type: ', type(key))
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
        fps = 1.0 / (time.time() - start_time)
        print("FPS: %.2f" % fps)
        # info = "time: %.2f ms" %(1000*(time.time() - start_time))
        #print(info)
        result = np.asarray(frame)
        result = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        # depth map을 칼라로 보기위함 
        # depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_frame, alpha=0.03), cv2.COLORMAP_JET)

        if not FLAGS.dont_show:
            cv2.imshow("Output Video", result)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            dc.release()
            cv2.destroyAllWindows()
            break                        

if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass