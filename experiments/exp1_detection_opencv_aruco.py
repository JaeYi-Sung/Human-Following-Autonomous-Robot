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

# Aruco marker
import cv2.aruco as aruco

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

def findArucoMarkers(img, markerSize=4, totalMarkers=250, draw=True):
    """
    :param img: image in which to find the aruco markers
    :param markerSize: the size of the markers
    :param totalMarkers: total number of markers that compose the dictionary
    :param draw: flag to draw bbox around markers detected
    :return: bounding boxes and id numbers of markers detected
    """
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Dinamically set markers to be detected
    key = getattr(aruco, f'DICT_{markerSize}X{markerSize}_{totalMarkers}')
    arucoDict = aruco.Dictionary_get(key)
    arucoParam = aruco.DetectorParameters_create()
    bboxs, ids, rejected = aruco.detectMarkers(imgGray, arucoDict, parameters=arucoParam)

    # Draw detected markers with bounding boxs
    if draw:
        aruco.drawDetectedMarkers(img, bboxs)
    
    # Return detected values for augmentation
    return [bboxs, ids]

# Function to label
def labelAruco(bbox, idx, img, drawId=True):
    """
    :param bbox: the four corner points of the box
    :param id: marker id of the corresponding box used only for display
    :param ig: the final image on which to draw
    :param drawId: flag to display the id of the detected markers
    :return: image with the labeled image
    """
    tl = int(bbox[0][0][0]), int(bbox[0][0][1])
    
    # print marker id
    if drawId:
        cv2.putText(img, str(idx), tl, cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)
    
    return img

def test_marker_detection(*test_type):
    
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
        
        arucoFound = findArucoMarkers(frame)
        
        if len(arucoFound[0])!=0:
            if int(FLAGS.marker_to_test) in arucoFound[1]:
                count += 1
                print(f"{image_file}: Detected")
            else: print(f"{image_file}: Undetected")
            
            for bbox, idx in zip(arucoFound[0], arucoFound[1]):
                frame = labelAruco(bbox, idx, frame)
        else:
            print(f"{image_file}: Undetected")
        
        frame_num +=1

        cv2.imshow("Output Video", frame)
        
        keyboard = cv2.waitKey(1) & 0xFF
        if keyboard == 27 or keyboard == 113: # ESC 또는 q 입력으로 프로그램 종료
            # dc.release()
            cv2.destroyAllWindows()
            print(f"key 'ESC' or 'q'---> End")
            break
        
    result = count / frame_num
    print(f"[Test Case] '{test_case}' ---> Detection Rate: {result}")
    print(f"[Test Case] '{test_case}' ---> End")    
    print("=============================================")
    
def main(_argv):
    del _argv
    
    # load configuration for object detector
    config = ConfigProto()
    config.gpu_options.allow_growth = True

    print("Detection Rate Experiment - OpenCV")
    
    for i in range(2): # 방향
        for j in range(3): # 간격
            test_marker_detection(i, j)
    test_marker_detection(2)
    
if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass