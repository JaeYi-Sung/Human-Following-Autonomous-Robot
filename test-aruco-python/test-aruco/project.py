import cv2
import cv2.aruco as aruco
import numpy as np
import os

import ArucoModule as arm

cap = cv2.VideoCapture(0)

augDic = arm.loadAugImages("./test-aruco-python/images")

while True:
    success, img = cap.read()
    arucoFound = arm.findArucoMarkers(img)

    # Loop through all the markers and augment each one
    if len(arucoFound[0])!=0:
        for bbox, id in zip(arucoFound[0], arucoFound[1]):
            if int(id) in augDic.keys():
                img = arm.augmentedAruco(bbox, id, img, augDic[int(id)]) # augment

    cv2.imshow("Image", img)
    cv2.waitKey(1)