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
from PIL import Image
import cv2, math, time
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.compat.v1 import ConfigProto
import json
import torch
import torchvision.transforms as transforms
import traitlets
import pickle

# trt_pose imports
sys.path.insert(1, '/home/ebrl/torch2trt/torch2trt')
from torch2trt import torch2trt, TRTModule
sys.path.insert(2, '/home/ebrl/torch2trt/trt_pose/trt_hand')
from preprocessdata_pe4 import Preprocessdata
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

flags.DEFINE_string('images', os.getenv('HOME') + '/Interaction_PE/images/right_fist/', 'path to images files') 
flags.DEFINE_string('results', os.getenv('HOME') + '/Interaction_PE/results/right_fist/', 'path to results files') 

# interaction을 위한 json 파일 load
with open('/home/ebrl/torch2trt/trt_pose/trt_hand/preprocess/hand_pose.json', 'r') as f:
    hand_pose = json.load(f)

with open('/home/ebrl/torch2trt/trt_pose/trt_hand/preprocess/gesture.json', 'r') as f:
    gesture = json.load(f)
gesture_type = gesture["classes"]
    
def main(_argv):
    del _argv
    # interaction을 위한 상수 설정
    WIDTH = 224
    HEIGHT = 224
    XCOOR = WIDTH * 2.857
    YCOOR = HEIGHT * 2.1429

    mean = torch.Tensor([0.485, 0.456, 0.406]).cuda()
    std = torch.Tensor([0.229, 0.224, 0.225]).cuda()
    device = torch.device('cuda')

    def preprocess(image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        image = transforms.functional.to_tensor(image).to(device)
        image.sub_(mean[:, None, None]).div_(std[:, None, None])
        return image[None, ...]

    def get_keypoints(objects, index, peaks):
        # check invalid index
        kpoint = []
        obj = objects[0][index]
        C = obj.shape[0]
        for j in range(C):
            k = int(obj[j])
            if k >= 0:
                peak = peaks[0][j][k] # peak[1]:width, peak[0]:height
                peak = (j, float(peak[0]), float(peak[1]))
                kpoint.append(peak)
                print('index:%d : success [%5.3f, %5.3f]'%(j, peak[1], peak[2]) )
            else:    
                peak = (j, None, None)
                kpoint.append(peak)
                print('index:%d : None'%(j) )
        return kpoint

    def draw_joints(frame, joints):
        count = 0
        width = frame.shape[1]
        height = frame.shape[0]
        for i in joints:
            if i==[0,0]:
                count+=1
        if count>= 3:
            return 
        for i in joints:
            cv2.circle(frame, (int(i[0] * XCOOR),int(i[1] * YCOOR)), 8, (148,0,211), 2)
        for i in hand_pose['skeleton']:
            if joints[i[0]-1][0]==0 or joints[i[1]-1][0] == 0:
                break
            cv2.line(frame, (joints[i[0]-1][0],joints[i[0]-1][1]), (joints[i[1]-1][0],joints[i[1]-1][1]), (106,90,205), 6)
		

    def execute_hand(img, frame):
        data = preprocess(img)
        cmap, paf = model_trt_hand(data)
        cmap, paf = cmap.detach().cpu(), paf.detach().cpu()
        counts, objects, peaks = parse_objects_hand(cmap, paf)
        joints = preprocessdata.joints_inference(img, counts, objects, peaks)
        #draw_joints(frame, joints)
        draw_objects_hand(frame, counts, objects, peaks)
        dist_bn_joints = preprocessdata.find_distance(joints)
        gesture = clf.predict([dist_bn_joints,[0]*num_parts_hand*num_parts_hand])
        gesture_joints = gesture[0]
        preprocessdata.prev_queue.append(gesture_joints)
        preprocessdata.prev_queue.pop(0)
        gesture_class = preprocessdata.print_label(frame, preprocessdata.prev_queue, gesture_type)
        return gesture_class

    topology_hand = coco.coco_category_to_topology(hand_pose)

    num_parts_hand = len(hand_pose['keypoints'])
    #num_links_hand = len(hand_pose['skeleton'])

    OPTIMIZED_MODEL_HAND  = '/home/ebrl/torch2trt/trt_pose/trt_hand/model/resnet18_baseline_att_244_244_trt.pth'

    model_trt_hand = TRTModule()

    model_trt_hand.load_state_dict(torch.load(OPTIMIZED_MODEL_HAND))
		
    parse_objects_hand = ParseObjects(topology_hand)
    draw_objects_hand = DrawObjects(topology_hand)
		
    preprocessdata = Preprocessdata(topology_hand, num_parts_hand)

    clf = pickle.load(open('/home/ebrl/torch2trt/trt_pose/svmmodel.sav', 'rb'))
    #i = 0

    pointing = 0
    palm = 0
    peace = 0
    ok = 0
    fist = 0
    no_gesture = 0

    for image_file in os.listdir(FLAGS.images):
        full_path = FLAGS.images + image_file
        frame = np.fromfile(full_path, np.uint8)
        frame = cv2.imdecode(frame, cv2.IMREAD_COLOR)
        image_data = cv2.resize(frame, (WIDTH, HEIGHT))

        gesture = execute_hand(image_data, frame)

        if gesture == 'pointing':
            pointing += 1
        elif gesture == 'palm':
            palm += 1
        elif gesture == 'peace':
            peace += 1
        elif gesture == 'ok':
            ok += 1
        elif gesture == 'fist':
            fist += 1
        else:
            no_gesture += 1

        if frame is None:
            print("Image load failed")
            return

        #cv2.imwrite(FLAGS.images + 'left_fist_' + str(i), frame)
        cv2.imwrite(FLAGS.results + image_file, frame)
        #i += 1

    print("=== Result ===")
    print("[test case] => " + FLAGS.images)
    print("pointing : ", pointing)
    print("palm : ", palm)
    print("peace : ", peace)
    print("ok : ", ok)
    print("fist: ", fist)
    print("no gesture: ", no_gesture)
    
if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
