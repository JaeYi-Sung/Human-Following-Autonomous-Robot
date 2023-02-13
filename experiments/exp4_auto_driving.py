#! /usr/bin/env python3

import os, sys

# 로봇 위치 받아오기
import rospy
from nav_msgs.msg import Odometry

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
from PIL import Image, ImageDraw
import cv2, math, time
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.compat.v1 import ConfigProto

# deep sort imports
from deep_sort.detection import Detection
from deep_sort import nn_matching, preprocessing
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet

# trt_pose imports
import json
import torch
import torchvision.transforms as transforms
import pickle
sys.path.insert(1, '/home/ebrl/torch2trt/torch2trt')
from torch2trt import TRTModule
sys.path.insert(2, '/home/ebrl/torch2trt/trt_pose/trt_hand')
sys.path.insert(3, '/home/ebrl/torch2trt/trt_pose/trt_pose')
from preprocessdata import Preprocessdata
# import coco
from parse_objects import ParseObjects
from draw_objects import DrawObjects
from datetime import datetime

# comment out below line to enable tensorflow logging outputs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0: tf.config.experimental.set_memory_growth(physical_devices[0], True)

# deepsort를 위한 변수 지정
flags.DEFINE_float('iou', 0.45, 'iou threshold')
flags.DEFINE_float('score', 0.50, 'score threshold')

YOLO_SIZE = 416
ORIGINAL_W = 640
ORIGINAL_H = 480

# path to weights file
weights_person = os.getenv('HOME') + '/wego_ws/src/scout_ros/scout_bringup/checkpoints/yolov4-tiny-416-person'
weights_marker = os.getenv('HOME') + '/wego_ws/src/scout_ros/scout_bringup/checkpoints/yolov4-tiny-416-marker'

# bounding box 색깔 지정
cmap = plt.get_cmap('tab20b')
colors = [cmap(i)[:3] for i in np.linspace(0, 1, 6)]
marker_colors = [[i * 255 for i in color] for color in colors[:5]]
person_color = [i * 255 for i in colors[5]]

# detection indexing
idx_to_name_marker = {0: "0", 1: "10", 2: "20", 3: "30", 4: "40"}
name_to_idx_marker = {"0": 0, "10": 1, "20": 2, "30": 3, "40": 4}
idx_to_name_person = {0: "person"}


# 로봇 위치 callback
# def callback(msg):
#     x = msg.pose.pose.position.x
#     y = msg.pose.pose.position.y
#     print(x, y)

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
    # original_h, original_w, _ = frame.shape
    bboxes = utils.format_boxes(bboxes, ORIGINAL_H, ORIGINAL_W)

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

# 손동작 인식을 위한 상수, 변수 설정
REST = 4 # 손동작 인식 후 인식 휴면 시간
TRT_POSE_SIZE = 224
# 추가
XCOOR = TRT_POSE_SIZE * 2.857 # 480 / 224
YCOOR = TRT_POSE_SIZE * 2.1429 # 640 / 224

NUM_KPOINT_HAND = 18
NUM_PARTS_HAND = 21
PHI = np.float64((1 + 5**0.5) / 2) # 1.618

gesture_type = ["no gesture", "palm"]
gesture_text_to_idx = {"no gesture": 0, "palm": 1}

mean = torch.Tensor([0.485, 0.456, 0.406]).cuda()
std = torch.Tensor([0.229, 0.224, 0.225]).cuda()
device = torch.device('cuda')

# make topologies (topology_human, toplogy_hand)
def coco_category_to_topology(coco_category):
    """Gets topology tensor from a COCO category
    """
    skeleton = coco_category['skeleton']
    K = len(skeleton)
    topology = torch.zeros((K, 4)).int()
    for k in range(K):
        topology[k][0] = 2 * k
        topology[k][1] = 2 * k + 1
        topology[k][2] = skeleton[k][0] - 1
        topology[k][3] = skeleton[k][1] - 1
    return topology

def preprocess(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image)
    image = transforms.functional.to_tensor(image).to(device)
    image.sub_(mean[:, None, None]).div_(std[:, None, None])
    return image[None, ...]

def get_keypoints(person_object, peaks, check_idx):
    # check invalid index
    kpoint = dict()
    for idx in range(NUM_KPOINT_HAND):
        if idx not in check_idx: continue
        k = int(person_object[idx])
        if k >= 0: kpoint[idx] = tuple(map(float, peaks[idx][k]))
    return kpoint

def execute_human(img, frame, model_trt_human, parse_objects_human, bbox):
    data = preprocess(img)
    cmap, paf = model_trt_human(data)
    cmap, paf = cmap.detach().cpu(), paf.detach().cpu()
    counts, objects, peaks = parse_objects_human(cmap, paf) #, cmap_threshold=0.15, link_threshold=0.15)
    check_idx = (17, 7, 8, 9, 10) # mid_shoulder, left_elbow, right_elbow, left_wrist, right_wrist)
    check_kp = dict() # 타깃 사람의 중요한 신체 부위
    bbox = [bbox[0] / ORIGINAL_W, bbox[1] / ORIGINAL_H, bbox[2] / ORIGINAL_W, bbox[3] / ORIGINAL_H]
    
    # bbox = bbox / ([ORIGINAL_W, ORIGINAL_H] * 2)
    for p_idx in range(counts[0]): # 사람 수 만큼
        keypoints = get_keypoints(objects[0][p_idx], peaks[0], check_idx) # p_idx 번째 사람
        if 17 not in keypoints: continue
        if (bbox[0] < keypoints[17][1] < bbox[2]) and (bbox[1] < keypoints[17][0] < bbox[3]):
            for key, val in keypoints.items():
                x = round(val[1] * XCOOR)
                y = round(val[0] * YCOOR)
                cv2.circle(frame, (x, y), 3, (0, 255, 0), 2)
                cv2.putText(frame , "%d" % int(key), (x + 5, y),  cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 1)
                check_kp[key] = (x, y)
            return check_kp
    return check_kp

# 큐에 넣을 때, 왼손 오른손 구분해서 넣기, 반환 값 설정하기
def execute_hand(img, frame, model_trt_hand, preprocessdata, parse_objects_hand, draw_objects_hand, clf):
    data = preprocess(img)
    cmap, paf = model_trt_hand(data)
    cmap, paf = cmap.detach().cpu(), paf.detach().cpu()
    counts, objects, peaks = parse_objects_hand(cmap, paf)
    joints = preprocessdata.joints_inference(img, counts, objects, peaks)
    draw_objects_hand(frame, counts, objects, peaks)                
    dist_bn_joints = preprocessdata.find_distance(joints)
    gesture = clf.predict([dist_bn_joints, [0] * NUM_PARTS_HAND * NUM_PARTS_HAND])
    gesture_joints = gesture[0]
    preprocessdata.prev_queue.append(gesture_joints)
    preprocessdata.prev_queue.pop(0)
    return gesture_text_to_idx[preprocessdata.print_label(frame, gesture_type)]

def crop_hand(image, elbow, wrist): # elbow, wrist의 좌표 x, y 순으로 입력
    # 넘파이로 변환
    elbow_vector = np.asarray(elbow)
    wrist_vector = np.asarray(wrist)

    # 손 위치 추정 계산
    lower_arm_vector = wrist_vector - elbow_vector
    hand_center = wrist + lower_arm_vector / PHI # 손이 팔보다 가까워서 두 배 더 크게 보임 (PHI * 2 / 2 대신 PHI)

    hand_half = np.max(np.abs(hand_center - elbow_vector))
    x_b, y_b = (hand_center - hand_half).astype(int)
    x_t, y_t = (hand_center + hand_half).astype(int)

    return image[max(0, y_b):min(ORIGINAL_H, y_t), max(0, x_b):min(ORIGINAL_W, x_t)]

def main(_argv):
    del _argv
    
    # initialize deep sort
    encoder = gdet.create_box_encoder(os.getenv('HOME') + '/wego_ws/src/scout_ros/scout_bringup/model_data/mars-small128.pb', batch_size=1)

    # calculate cosine distance metric and initialize tracker
    metric_person = nn_matching.NearestNeighborDistanceMetric("cosine", 0.4, None)
    tracker_person = Tracker(metric_person)

    # load configuration for object detector
    config = ConfigProto()
    config.gpu_options.allow_growth = True

    saved_model_loaded_person = tf.saved_model.load(weights_person, tags=[tag_constants.SERVING])
    infer_person = saved_model_loaded_person.signatures['serving_default']

    saved_model_loaded_marker = tf.saved_model.load(weights_marker, tags=[tag_constants.SERVING])
    infer_marker = saved_model_loaded_marker.signatures['serving_default']

    # load & save trt_pose model (model_trt_human, model_trt_hand)
    model_trt_human = TRTModule()
    model_trt_hand = TRTModule()
    OPTIMIZED_MODEL_HUMAN = os.getenv('HOME') + '/torch2trt/trt_pose/tasks/human_pose/resnet18_baseline_att_224x224_A_epoch_249_trt.pth'
    OPTIMIZED_MODEL_HAND = os.getenv('HOME') + '/torch2trt/trt_pose/trt_hand/model/resnet18_baseline_att_244_244_trt.pth'
    model_trt_human.load_state_dict(torch.load(OPTIMIZED_MODEL_HUMAN)) # load
    model_trt_hand.load_state_dict(torch.load(OPTIMIZED_MODEL_HAND)) # load
    
    human_pose = json.load(open(os.getenv('HOME') + '/torch2trt/trt_pose/tasks/human_pose/human_pose.json', 'r'))
    hand_pose = json.load(open(os.getenv('HOME') + '/torch2trt/trt_pose/trt_hand/preprocess/hand_pose.json', 'r'))
    clf = pickle.load(open('/home/ebrl/torch2trt/trt_pose/svmmodel.sav', 'rb'))
    
    topology_human = coco_category_to_topology(human_pose)
    topology_hand = coco_category_to_topology(hand_pose)
    
    preprocessdata_left = Preprocessdata(topology_hand, NUM_PARTS_HAND)
    preprocessdata_right = Preprocessdata(topology_hand, NUM_PARTS_HAND)
    
    parse_objects_human = ParseObjects(topology_human)
    parse_objects_hand = ParseObjects(topology_hand)
    draw_objects_hand = DrawObjects(topology_hand)
    
    # Depth camera class 불러오기
    dc = DepthCamera()

    # 장애물 영역 기본값 받아오기
    default = Default_dist()

    # 로봇 위치 subscribe
    # rospy.init_node('odom_sub')
    # sub = rospy.Subscriber('/odom', Odometry, callback)

    # ROS class init
    go = scout_pub_basic()
    rate = rospy.Rate(60)
    
    # target 설정을 위한 변수
    target_marker = "0"
    target_changed = True
    target_person_id = None
    target_lost_track_id = True
    
    # 로봇 모터 제어를 위한 초깃값 설정
    x, y, z, th, speed, turn, key = 0, 0, 0, 0, 0.1, 0.1, ''
    
    # 주행 상태 설정 변수
    inactive_time = 0
    mode = 1 # 주행 상태 (1: 주행 상태, 0: 정지 상태)
      
    # 프레임 시작 시간 측정
    previous_time = time.time()
    frame_num = 0
    
    # while video is running
    while not rospy.is_shutdown():

        # depth camera 사용
        return_value, depth_frame, frame = dc.get_frame()

        if return_value:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            input_image = frame.copy()
        else:
            print('Video has ended or failed, try a different video format!')
            break
        
        # 프레임 넘버 1 증가
        frame_num +=1
        print('Frame #: ', frame_num)
        
        # calculate frames per second of running detections
        now_time = time.time()
        fps = 1.0 / (now_time - previous_time)
        previous_time = now_time
        print("FPS: %.2f" % fps)
        
        # 장애물 회피를 위한 ROI 디폴트 세팅하기 (현재는 10프레임만) 추가
        if frame_num < 11 :
            default.default_update(depth_frame)
            continue

        # 프레임 이미지 정보
        image_data = cv2.resize(input_image, (YOLO_SIZE, YOLO_SIZE))
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

        # <st-mini 제어를 위한 Publisher code>
        go.update(x, y, z, th, speed, turn)
        
        lost = True # tracker.lost가 True라면 Target lost
        if not tracker_person.lost:
            # 추적 알고리즘
            lost = True # 트래킹할 객체 중 타깃이 없다면 True, 아니면 False

            # 추적할 객체가 있다면 동작
            for track in tracker_person.tracks:

                if not track.is_confirmed() or track.time_since_update > 1: continue
                
                bbox = track.to_tlbr()
                track_id = track.track_id
                
                # target marker가 변한 경우, track.id 설정 후 그 사람 tracking
                # 아니라면, 기존의 track.id를 가진 사람 tracking
                if target_changed:
                    for marker_bbox in target_marker_bboxes:
                        if marker_bbox[0] >= bbox[0] and marker_bbox[1] >= bbox[1] and marker_bbox[2] <= bbox[2] and marker_bbox[3] <= bbox[3]:
                            x, y, z, th, speed, turn, key = 0, 0, 0, 0.1, 0.1, '' # 로봇 모터 제어를 위한 초깃값 설정
                            target_person_id = track_id
                            target_changed = False
                            target_lost_track_id = False
                            print(f"[{datetime.now().time()}] Track ID '{track_id}' Set to Target Person.")
                            break

                elif target_lost_track_id:
                    for marker_bbox in target_marker_bboxes:
                        if marker_bbox[0] >= bbox[0] and marker_bbox[1] >= bbox[1] and marker_bbox[2] <= bbox[2] and marker_bbox[3] <= bbox[3]:
                            target_person_id = track_id
                            target_lost_track_id = False
                            print(f"[{datetime.now().time()}] Missed Track ID '{track_id}' Reset.")
                            break

                if target_changed: continue # target marker 바뀌고 다시 track id 설정 안된 경우, track 하지 않기
                if track_id != target_person_id: continue  # target id에 해당하지 않은 사람 객체 무시
                
                # target id에 해당하는 사람 객체 tracking            
                lost = False
                
                # draw target person's bbox on screen
                cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), person_color, 2)
                cv2.rectangle(frame, (int(bbox[0]), int(bbox[1]-30)), (int(bbox[0])+85, int(bbox[1])), person_color, -1)
                cv2.putText(frame, "Person-" + str(track_id), (int(bbox[0]), int(bbox[1]-10)), 0, 0.75, (255,255,255), 2)
                
                if inactive_time and now_time - inactive_time >= REST: # 손동작 인식 휴면시간 초과
                    print(f"[{datetime.now().time()}] Hand Recognition Dormant Timeout")
                    inactive_time = 0 # 초기화
                elif not inactive_time: # 손동작 인식 활성시간 
                    image_data = cv2.resize(input_image, (TRT_POSE_SIZE, TRT_POSE_SIZE))
                    check_kp = execute_human(image_data, frame, model_trt_human, parse_objects_human, bbox)

                    gesture_idx = 0 # no gesture 로 초기화
                    if 7 in check_kp and 9 in check_kp: # left hand
                        cropped_image = crop_hand(input_image, check_kp[7], check_kp[9])
                        cropped_image = cv2.rotate(cropped_image, cv2.ROTATE_180)
                        cropped_image = cv2.resize(cropped_image, (TRT_POSE_SIZE, TRT_POSE_SIZE))
                        gesture_idx = max(gesture_idx, execute_hand(cropped_image, frame, model_trt_hand, preprocessdata_left, parse_objects_hand, draw_objects_hand, clf))
                        
                    if gesture_idx == 0 and 8 in check_kp and 10 in check_kp: # right hand
                        cropped_image = crop_hand(input_image, check_kp[8], check_kp[10])
                        cropped_image = cv2.rotate(cropped_image, cv2.ROTATE_180)
                        cropped_image = cv2.resize(cropped_image, (TRT_POSE_SIZE, TRT_POSE_SIZE))
                        gesture_idx = max(gesture_idx, execute_hand(cropped_image, frame, model_trt_hand, preprocessdata_right, parse_objects_hand, draw_objects_hand, clf))

                    if gesture_idx: # palm (주행 상태 토글)
                        print(f"[{datetime.now().time()}] Palm Recognition")
                        mode = 1 - mode
                        inactive_time = now_time
                        print(f"Drive Mode: {mode} -> {1 - mode} (Stop: 0, Drive: 1) & Hand Recognition Dormant Start")

                # target person의 bbox의 w, h, cx, cy 계산
                w, h = int(bbox[2]-bbox[0]), int(bbox[3]-bbox[1])
                cx, cy = int(w/2 + bbox[0]), int(h/2 + bbox[1])
                
                # 사람과 로봇의 거리: person_distance
                person_distance = person_dist(depth_frame, cx, cy, h)
                print('Person Distance: ', person_distance)
                
                go.sendMsg(1)
                
                # mode가 1이라면 주행, 0 이라면 비주행
                if mode: key, speed, turn = drive(cx / ORIGINAL_W, key, turn, speed, person_distance)
                else: key = 'stop'
                
                break # 타깃 사람 찾으면 break
        if lost: 
            go.sendMsg(frame_num % 2)
            target_person_id = None
            target_lost_track_id = True
            key = 'stop'
            print('There are no objects to track.')
            
        # 주행 알고리즘(drive)를 거치고 나온 속도/방향을 로봇에 전달
        x, y, z, th, speed, turn = key_move(key, x, y, z, th, speed, turn)

        print(f'key: {key}')
        print(f'x: {x}, y: {y}, th: {th}, speed: {speed}, turn: {turn}')

        # 로봇 현재 위치 출력
        odom_sub = rospy.wait_for_message('/odom', Odometry)
        pos_x = odom_sub.pose.pose.position.x
        pos_y = odom_sub.pose.pose.position.y
        print(f"robot's position: ({pos_x}, {pos_y})")
        print()

        # ROS Rate sleep
        rate.sleep()

        result = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.imshow("Output Video", result)
                    
        keyboard = cv2.waitKey(1) & 0xFF
        if keyboard == 27 or keyboard == 113: # ESC 또는 q 입력으로 프로그램 종료
            dc.release()
            cv2.destroyAllWindows()
            print(f"[{datetime.now().time()}] Keyboard Input '{chr(keyboard)}': End")
            break
        elif 48 <= keyboard <= 52: # 0, 1, 2, 3, 4 키보드 입력으로 타겟 마커 변경
            target_marker = idx_to_name_marker[keyboard - 48]
            print(f"[{datetime.now().time()}] Keyboard Input '{chr(keyboard)}': Target Marker '{target_marker}'")
            target_person_id = None
            target_changed = True
        
if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass