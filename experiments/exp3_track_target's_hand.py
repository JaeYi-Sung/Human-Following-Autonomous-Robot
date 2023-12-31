#! /usr/bin/env python3

import os, sys

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
import traitlets
import pickle
sys.path.insert(1, '/home/ebrl/torch2trt/torch2trt')
from torch2trt import torch2trt, TRTModule
sys.path.insert(2, '/home/ebrl/torch2trt/trt_pose/trt_hand')
from preprocessdata_exp2 import Preprocessdata
sys.path.insert(3, '/home/ebrl/torch2trt/trt_pose/trt_pose')
import coco
import models
from resnet import resnet18_baseline_att
from parse_objects import ParseObjects
from draw_objects import DrawObjects

# comment out below line to enable tensorflow logging outputs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0: tf.config.experimental.set_memory_growth(physical_devices[0], True)

# deepsort를 위한 변수 지정
flags.DEFINE_string('weights_person', os.getenv('HOME') + '/wego_ws/src/scout_ros/scout_bringup/checkpoints/yolov4-tiny-416-person', 'path to weights file')
flags.DEFINE_string('weights_marker', os.getenv('HOME') + '/wego_ws/src/scout_ros/scout_bringup/checkpoints/yolov4-tiny-416-marker', 'path to weights file')
flags.DEFINE_integer('size', 416, 'resize images to')
flags.DEFINE_float('iou', 0.45, 'iou threshold')
flags.DEFINE_float('score', 0.50, 'score threshold')
flags.DEFINE_boolean('count', False, 'count objects being tracked on screen')
flags.DEFINE_string('images', os.getenv('HOME') + '/EXP2_230118/images/right_palm_front/', 'input images path')
flags.DEFINE_string('results', os.getenv('HOME') + '/EXP2_230118/results/right_palm_front/', 'output images path')

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

########### Interaction ############
# interaction을 위한 상수 설정
TRT_POSE_SIZE = 224
XCOOR = TRT_POSE_SIZE * 2.857
YCOOR = TRT_POSE_SIZE * 2.1429

NUM_PARTS_HAND = 21

PHI = (1 + 5**0.5) / 2 # 1.618
D = 2 # 손이 팔보다 가까워서 더 크게 보임

human_pose = json.load(open(os.getenv('HOME') + '/torch2trt/trt_pose/tasks/human_pose/human_pose.json', 'r'))
hand_pose = json.load(open(os.getenv('HOME') + '/torch2trt/trt_pose/trt_hand/preprocess/hand_pose.json', 'r'))
clf = pickle.load(open('/home/ebrl/torch2trt/trt_pose/svmmodel.sav', 'rb'))

gesture_type = ["palm","no gesture"]
gesture_text_to_idx = {"palm": 1, "no gesture": 0}

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
    for idx in range(person_object.shape[0]):
        if idx not in check_idx: continue # 살펴볼 keypoint가 아니면 무시
        k = int(person_object[idx])
        if k >= 0: kpoint[idx] = tuple(map(float, peaks[idx][k]))
        #print(f"idx: {idx}, kpoint: {kpoint}")
    return kpoint

def execute_human(img, frame, bbox_frame, model_trt_human, parse_objects_human, bbox):
    #print("execute_human function - bbox: ", bbox)
    color = (0, 255, 0)
    data = preprocess(img)
    cmap, paf = model_trt_human(data)
    cmap, paf = cmap.detach().cpu(), paf.detach().cpu()
    counts, objects, peaks = parse_objects_human(cmap, paf) #, cmap_threshold=0.15, link_threshold=0.15)
    #check_idx = (17, 7, 8, 9, 10) # mid_shoulder, left_elbow, right_elbow, left_wrist, right_wrist)
    check_idx = (17, 8, 10)
    #target_kp = None
    check_kp = dict() # 모든 사람의 중요한 신체 부위
    bbox = [bbox[0] / 640, bbox[1] / 480, bbox[2] / 640, bbox[3] / 480]
    midbbox = (bbox[0] + bbox[2]) / 2
    min_dist = 123456789
    target_idx = 0

    for p_idx in range(counts[0]): # 사람 수 만큼
        #print(f"counts[0]: {counts[0]}")
        keypoints = get_keypoints(objects[0][p_idx], peaks[0], check_idx) # p_idx 번째 사람
        mid2shou = 0
    
        if 17 not in keypoints: continue
        else:
            mid2shou = abs(midbbox - keypoints[17][1])
            if mid2shou < min_dist:
                min_dist = mid2shou
                target_idx = p_idx

    target_kp = get_keypoints(objects[0][target_idx], peaks[0], check_idx)
    if (bbox[0] < target_kp[17][1] < bbox[2]) and (bbox[1] < target_kp[17][0] < bbox[3]):
        for key, val in target_kp.items():
            x = round(val[1] * XCOOR)
            y = round(val[0] * YCOOR)
            cv2.circle(bbox_frame, (x, y), 3, color, 2)
            cv2.putText(bbox_frame, "%d" % int(key), (x + 5, y),  cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 1)
            check_kp[key] = (x, y)
        return check_kp
    return check_kp


# 큐에 넣을 때, 왼손 오른손 구분해서 넣기, 반환 값 설정하기
def execute_hand(img, frame, model_trt_hand, preprocessdata, parse_objects_hand, draw_objects_hand):
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
    hand_center = wrist + lower_arm_vector / (PHI / D * 2)

    hand_half = np.max(np.abs(hand_center - elbow_vector))
    x_b, y_b = (hand_center - hand_half).astype(int)
    x_t, y_t = (hand_center + hand_half).astype(int)

    h, w, _ = image.shape
    cropped_image = image[max(0, y_b):min(h, y_t), max(0, x_b):min(w, x_t)]

    return cropped_image

def main(_argv):
    del _argv
    
    # target 설정을 위한 변수
    target_marker = "20"

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

    # load & save trt_pose model (model_trt_human, model_trt_hand)
    model_trt_human = TRTModule()
    model_trt_hand = TRTModule()
    OPTIMIZED_MODEL_HUMAN = os.getenv('HOME') + '/torch2trt/trt_pose/tasks/human_pose/resnet18_baseline_att_224x224_A_epoch_249_trt.pth'
    OPTIMIZED_MODEL_HAND = os.getenv('HOME') + '/torch2trt/trt_pose/trt_hand/model/resnet18_baseline_att_244_244_trt.pth'
    model_trt_human.load_state_dict(torch.load(OPTIMIZED_MODEL_HUMAN)) # load
    model_trt_hand.load_state_dict(torch.load(OPTIMIZED_MODEL_HAND)) # load
    
    topology_human = coco_category_to_topology(human_pose)
    topology_hand = coco_category_to_topology(hand_pose)
    
    preprocessdata_left = Preprocessdata(topology_hand, NUM_PARTS_HAND)
    preprocessdata_right = Preprocessdata(topology_hand, NUM_PARTS_HAND)
    
    parse_objects_human = ParseObjects(topology_human)
    parse_objects_hand = ParseObjects(topology_hand)
    draw_objects_hand = DrawObjects(topology_hand)
    
    frame_num = 0
    count = 0
    
    for image_file in os.listdir(FLAGS.images):
        
        frame_num += 1
        gesture_idx = 0
        #print(f"image_file: {image_file}")

        full_path = FLAGS.images + image_file
        frame = np.fromfile(full_path, np.uint8)
        frame = cv2.imdecode(frame, cv2.IMREAD_COLOR)
        bbox_frame = frame.copy()
#        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # 프레임 이미지 정보
        image_data = cv2.resize(frame, (FLAGS.size, FLAGS.size))
        image_data = image_data / 255.
        image_data = image_data[np.newaxis, ...].astype(np.float32)
        
        batch_data = tf.constant(image_data)
        
        target_marker_bboxes = []
        detections_marker = detect_objects(infer_marker, batch_data, frame, encoder, idx_to_name_marker)
         # draw bbox for all detections and find the target marker's bbox
        target_marker_bboxes = [detection.to_tlbr() for detection in detections_marker if detection.get_class() == target_marker]
        #print("target_marker_bboxes: ", target_marker_bboxes)
        # for bbox in target_marker_bboxes:
            # color = marker_colors[name_to_idx_marker[target_marker]]
            # cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
            # cv2.rectangle(frame, (int(bbox[0]), int(bbox[1]-30)), (int(bbox[0])+34, int(bbox[1])), color, -1)
            # cv2.putText(frame, target_marker, (int(bbox[0]), int(bbox[1]-10)), 0, 0.75, (255,255,255), 2)

        # target marker를 가진 사람 detection 후 tracking
        detections_person = detect_objects(infer_person, batch_data, frame, encoder, idx_to_name_person)
        
        # Call the tracker
 #        tracker_person.predict()
 #        tracker_person.update(detections_person, frame_num)

        # 추적할 객체가 있다면 동작
        #print("detections_person length: ", len(detections_person))
        #print("frame.shape: ", frame.shape)
        for detection in detections_person:

            bbox = detection.to_tlbr()
            for marker_bbox in target_marker_bboxes:
                if marker_bbox[0] >= bbox[0] and marker_bbox[1] >= bbox[1] and marker_bbox[2] <= bbox[2] and marker_bbox[3] <= bbox[3]:
                    #print("There is a target person.")
                    # draw target person's bbox on screen
                    cv2.rectangle(bbox_frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), person_color, 2)
                    cv2.rectangle(bbox_frame, (int(bbox[0]), int(bbox[1]-30)), (int(bbox[0])+85, int(bbox[1])), person_color, -1)
                    cv2.putText(bbox_frame, "Person", (int(bbox[0]), int(bbox[1]-10)), 0, 0.75, (255,255,255), 2)
                    cv2.imwrite(FLAGS.results + image_file, bbox_frame)
#                    cv2.imwrite(FLAGS.results + 'frame' + image_file, frame)

                    ##### Interaction #####
                    image_data = cv2.resize(frame, (TRT_POSE_SIZE, TRT_POSE_SIZE))
                    check_kp = execute_human(image_data, frame, bbox_frame, model_trt_human, parse_objects_human, bbox)
                    cv2.imwrite(FLAGS.results + image_file, bbox_frame)
                    
                    # if 7 in check_kp and 9 in check_kp: # left arm
                    #     cropped_frame = crop_hand(frame, check_kp[7], check_kp[9])
                    #     cropped_frame = cv2.rotate(cropped_frame, cv2.ROTATE_180)
                    #     # cropped_frame = cv2.cvtColor(cropped_frame, cv2.COLOR_RGB2BGR)
                    #     cropped_frame = cv2.resize(cropped_frame, (TRT_POSE_SIZE, TRT_POSE_SIZE))
                    #     gesture_idx = max(gesture_idx, execute_hand(cropped_frame, frame, model_trt_hand, preprocessdata_left, parse_objects_hand, draw_objects_hand)) # 리턴 값 받기, 인수 변경
                    #     cropped_result = np.asarray(cropped_frame)
                    #     # cropped_result = cv2.cvtColor(cropped_frame, cv2.COLOR_RGB2BGR)
                    #     cv2.imwrite(FLAGS.results + os.path.splitext(image_file)[0] + '_cropped_left' + os.path.splitext(image_file)[1], cropped_result)

                    if gesture_idx == 0 and 8 in check_kp and 10 in check_kp: # right arm
                    # if 8 in check_kp and 10 in check_kp:
                        cropped_frame = crop_hand(frame, check_kp[8], check_kp[10])
                        cropped_frame = cv2.rotate(cropped_frame, cv2.ROTATE_180)
                        # cropped_frame = cv2.cvtColor(cropped_frame, cv2.COLOR_RGB2BGR)
                        cropped_frame = cv2.resize(cropped_frame, (TRT_POSE_SIZE, TRT_POSE_SIZE))
                        #print("right - execute_hand")
                        gesture_idx = max(gesture_idx, execute_hand(cropped_frame, frame, model_trt_hand, preprocessdata_right, parse_objects_hand, draw_objects_hand)) # 리턴 값 받기, 인수 변경
                        cropped_result = np.asarray(cropped_frame)
                        # cropped_result = cv2.cvtColor(cropped_frame, cv2.COLOR_RGB2BGR)
                        cv2.imwrite(FLAGS.results + os.path.splitext(image_file)[0] + '_cropped_right' + os.path.splitext(image_file)[1], cropped_result)
                        count += gesture_idx
                    print(f'frame: {frame_num}, count: {count}')
                    break
        
    print("\n=============================")
    print('[test case] => : ', FLAGS.images)
    print("palm : ", count)
    print("no gesture : ", frame_num - count)

        
if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass