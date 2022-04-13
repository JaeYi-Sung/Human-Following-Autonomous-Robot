# RealTime-Object-Detection

### Development Environment
- Ubuntu Linux v18.04 - JetPack 4.3
- CUDA 10.2
- cuDNN 7.6.5
- nvidia driver 470
- ROS Melodic
- OpenCV 4.2
- Tensorflow 2.3.1
- Jupyter Lab


### Techs that we're going to use
<p align="center">
  <img src="https://img.shields.io/badge/Pytorch-EE4C2C?style=flat-square&logo=Pytorch&logoColor=white"/></a> &nbsp     
  <img src="https://img.shields.io/badge/Tensorflow-FF6F00?style=flat-square&logo=Tensorflow&logoColor=white"/></a> &nbsp     
  <img src="https://img.shields.io/badge/YOLOv4-00FFFF?style=flat-square&logo=YOLO&logoColor=white"/></a> &nbsp   
  <img src="https://img.shields.io/badge/OpenCV-5C3EE8?style=flat-square&logo=OpenCV&logoColor=white"/></a> &nbsp
  <img src="https://img.shields.io/badge/Ubuntu-E95420?style=flat-square&logo=Ubuntu&logoColor=white"/></a> &nbsp 
  <img src="https://img.shields.io/badge/ROS-22314E?style=flat-square&logo=ROS&logoColor=white"/></a> &nbsp 
</p>

### Process
- [x] 개발환경 구축
- [x] 알고리즘 자료 조사
- [x] ArUco markers 데이터셋 만들기
  - [x] 이미지 촬영하기
  - [x] 이미지 라벨링
- [x] weights 만들기
  - [x] ArUco markers Dataset YOLOv5 모델 학습
  - [x] Person Dataset과 ArUco markers Dataset 합쳐서 Yolov5 모델 학습
  - [x] 성능 향상 공부
- [x] 계획 수정: YOLOv5 에서 YOLOv4로 변경, Person Dataset 변경, OpenCV의 aruco 모듈 적용
- [x] Person Dataset: MobilityAids 사용
- [x] OpenCV aruco 모듈 적용
- [x] 계획 수정: Person Dataset과 ArUco markers Dataset 합쳐서 YOLOv4 모델 학습
- [x] ArUco markers YOLOv4 학습
- [ ] weights 만들기
  - [x] Yolov4 개발 환경 구축
  - [x] Person Dataset 만들기
  - [x] Person Dataset Yolov4 모델 학습
  - [x] Person Dataset과 ArUco marker Dataset Yolov4 모델 학습
- [ ] 특정 ArUco marker를 지닌 사용자 스위칭 기능
