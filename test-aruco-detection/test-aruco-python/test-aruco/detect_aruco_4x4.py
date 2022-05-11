import cv2
import cv2.aruco as aruco
import numpy as np
import os

# Function to detect aruco markers
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
def labelAruco(bbox, id, img, drawId=True):
    """
    :param bbox: the four corner points of the box
    :param id: marker id of the corresponding box used only for display
    :param ig: the final image on which to draw
    :param drawId: flag to display the id of the detected markers
    :return: image with the labeled image
    """
    tl = bbox[0][0][0], bbox[0][0][1]
    tr = bbox[0][1][0], bbox[0][1][1]
    br = bbox[0][2][0], bbox[0][2][1]
    bl = bbox[0][3][0], bbox[0][3][1]

    # print marker id
    if drawId:
        cv2.putText(img, str(id), tl, cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)
    
    return img

def main():
    cap = cv2.VideoCapture(0)

    while True: # 화면을 끌 때 CTRL+C 외의 방법으로 끄는 방법 찾기
        success, img = cap.read()
        arucoFound = findArucoMarkers(img)

        # Loop through all the markers and augment each one
        if len(arucoFound[0])!=0:
            for bbox, id in zip(arucoFound[0], arucoFound[1]):
                img = labelAruco(bbox, id, img)

        cv2.imshow("Image", img)
        cv2.waitKey(1)
    
if __name__ == "__main__":
    main()
