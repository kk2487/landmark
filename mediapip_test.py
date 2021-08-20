import os
import cv2
import mediapipe as mp
import numpy as np
import sys
import time
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
import PyQt5.QtGui as QtGui
from PyQt5.QtCore import *
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
mp_holistic = mp.solutions.holistic

def read_classes(file_path):

	fp = open(file_path, "r")
	classes = fp.readline()
	classes = classes. split(",")
	fp.close()

	return classes

class Qt(QWidget):
	def mv_Chooser(self):    
		opt = QFileDialog.Options()
		opt |= QFileDialog.DontUseNativeDialog
		fileUrl = QFileDialog.getOpenFileName(self,"Input Video", "C:/Users/hongze/Desktop/night/crop","Mp4 (*.mp4)", options=opt)
	
		return fileUrl[0]
		
# For webcam input:
qt_env = QApplication(sys.argv)
process = Qt()
fileUrl = process.mv_Chooser()
cap = cv2.VideoCapture(fileUrl)

#videoWriter = cv2.VideoWriter("./result.avi",cv2.VideoWriter_fourcc('X','V','I','D'),30,(920,720))

img_n = 0

num_frame = 10
imgSize = (224,224)

ret, frame = cap.read()
height, width = frame.shape[:2]
height, width = frame.shape[:2]

with mp_holistic.Holistic(
	min_detection_confidence=0.5,
		min_tracking_confidence=0.5) as holistic:
	while (ret):
		ret, image = cap.read()
		if not ret:
			print("Ignoring empty camera frame.")
	  # If loading a video, use 'break' instead of 'continue'.
			break
		start = time.time()
		# Flip the image horizontally for a later selfie-view display, and convert
		# the BGR image to RGB.
		image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		# To improve performance, optionally mark the image as not writeable to
		# pass by reference.
		image.flags.writeable = False
		results = holistic.process(image)

		# Draw landmark annotation on the image.
		image.flags.writeable = True
		image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
		mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACE_CONNECTIONS)
		mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
		mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
		mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
		
		# 提取特徵點座標 (如果有關節點偵測結果)
		face_res = []
		res = []
		try :
			face_res = [[data_point.x,data_point.y] for data_point in results.face_landmarks.landmark]
			face_res = np.array(face_res, dtype=np.float32)
		except AttributeError:
			print("THIS FRAME HAS NO Detection")

		try :
			res = [[data_point.x, data_point.y, data_point.z, data_point.visibility] for data_point in results.pose_landmarks.landmark]
		except AttributeError:
			print("THIS FRAME HAS NO Detection")

		# 繪製關節圖供動作分類使用
		j_width, j_height = 224, 224
		joint_result = np.zeros((j_height, j_width), np.uint8)
		
		try :
			for i in range (len(face_res)):
				cv2.circle(joint_result,(int(face_res[i][0]*j_width), int(face_res[i][1]*j_height)), 1, 255, -1)
		except AttributeError:
			print("ERROR")

		try :
			for i in range (len(res)):
				x = int(res[i][0]*width)
				y = int(res[i][1]*height)				
		except AttributeError:
			print("ERROR")

		# 計算左右手掌中心點

		x_l = 0
		y_l = 0
		x_r = 0
		y_r = 0

		if(int(len(res))>=33):
			#16,18,20,22 right hand
			x_l = int( (res[16][0]+res[18][0]+res[20][0]+res[22][0])/4*j_width )
			y_l = int( (res[16][1]+res[18][1]+res[20][1]+res[22][1])/4*j_height )
			#15,17,19,21 left hand
			x_r = int( (res[15][0]+res[17][0]+res[19][0]+res[21][0])/4*j_width )
			y_r = int( (res[15][1]+res[17][1]+res[19][1]+res[21][1])/4*j_height )

		# 繪製左右手掌區域
		cv2.circle(joint_result,(x_r, y_r), 10, 255, -1)
		cv2.circle(joint_result,(x_l, y_l), 10, 255, -1)

		# 繪製身體關節點連線
		if(int(len(res))>=33):
			cv2.line(joint_result, (int(res[12][0]*j_width),int(res[12][1]*j_height)), (int(res[11][0]*j_width),int(res[11][1]*j_height)), 255, 2)
			cv2.line(joint_result, (int(res[11][0]*j_width),int(res[11][1]*j_height)), (int(res[23][0]*j_width),int(res[23][1]*j_height)), 255, 2)
			cv2.line(joint_result, (int(res[23][0]*j_width),int(res[23][1]*j_height)), (int(res[24][0]*j_width),int(res[24][1]*j_height)), 255, 2)
			cv2.line(joint_result, (int(res[24][0]*j_width),int(res[24][1]*j_height)), (int(res[12][0]*j_width),int(res[12][1]*j_height)), 255, 2)

			cv2.line(joint_result, (int(res[11][0]*j_width),int(res[11][1]*j_height)), (int(res[13][0]*j_width),int(res[13][1]*j_height)), 255, 2)
			cv2.line(joint_result, (int(res[12][0]*j_width),int(res[12][1]*j_height)), (int(res[14][0]*j_width),int(res[14][1]*j_height)), 255, 2)

			cv2.line(joint_result, (int(res[13][0]*j_width),int(res[13][1]*j_height)), (x_r,y_r), 255, 2)
			cv2.line(joint_result, (int(res[14][0]*j_width),int(res[14][1]*j_height)), (x_l,y_l), 255, 2)

		joint_result = cv2.resize(joint_result, imgSize)
		cv2.imshow("image", image)
		cv2.imshow("joint_result", joint_result)
		#videoWriter.write(image)
		if cv2.waitKey(1) & 0xFF == 27:
			#videoWriter.release()
			break
	cap.release()
	#videoWriter.release()
"""
with mp_pose.Pose(
	upper_body_only = True, 
	min_detection_confidence=0.95,
	min_tracking_confidence=0.95) as pose:
	while cap.isOpened():
		success, image = cap.read()
		if not success:
			print("Ignoring empty camera frame.")
			# If loading a video, use 'break' instead of 'continue'.
			break
		start = time.time()
		# Flip the image horizontally for a later selfie-view display, and convert
		# the BGR image to RGB.
		image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		# To improve performance, optionally mark the image as not writeable to
		# pass by reference.
		image.flags.writeable = False
		results = pose.process(image)

		# Draw the pose annotation on the image.
		image.flags.writeable = True
		image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
		mp_drawing.draw_landmarks(
			image, results.pose_landmarks, mp_pose.UPPER_BODY_POSE_CONNECTIONS )
		end = time.time()
		# print(results.pose_landmarks)
		
		res = [[data_point.x, data_point.y, data_point.z, data_point.visibility] for data_point in results.pose_landmarks.landmark]
		width, height = 300, 400
		result = np.zeros((height, width), np.uint8)
		#print(len(res))
		#16,18,20,22 right hand
		x_r = int( (res[16][0]+res[18][0]+res[20][0]+res[22][0])/4*width )
		y_r = int( (res[16][1]+res[18][1]+res[20][1]+res[22][1])/4*height )
		#15,17,19,21 left hand
		x_l = int( (res[15][0]+res[17][0]+res[19][0]+res[21][0])/4*width )
		y_l = int( (res[15][1]+res[17][1]+res[19][1]+res[21][1])/4*height )

		#cv2.circle(result,(x_r, y_r), 15, 255, -1)
		#cv2.circle(result,(x_l, y_l), 15, 255, -1)
		
		for i in range (len(res)):
			x = int(res[i][0]*width)
			y = int(res[i][1]*height)
			cv2.circle(result,(x, y), 2, 255, -1)
			print(x, y)
		
		print("-------------------------------------------")
		print(1/(end-start))
		cv2.imshow('MediaPipe Pose', image)
		cv2.imshow('result', result)
		videoWriter.write(image)
		if cv2.waitKey(5) & 0xFF == 27:
			videoWriter.release()
			break
	videoWriter.release()
	cap.release()
"""