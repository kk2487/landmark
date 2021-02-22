import os
import sys
import cv2 
import torch
import torchvision
import numpy as np
import time
from tqdm import tqdm
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
import PyQt5.QtGui as QtGui
from PyQt5.QtCore import *
from pathlib import Path  

class Qt(QWidget):
	def mv_Chooser(self):    
		opt = QFileDialog.Options()
		opt |= QFileDialog.DontUseNativeDialog
		fileUrl = QFileDialog.getOpenFileName(self,"Input Video", "C:/Users/hongze/Desktop/","Mp4 (*.mp4)", options=opt)
	
		return fileUrl[0]

if __name__ == '__main__':
	model = torch.hub.load('ultralytics/yolov5', 'custom', path_or_model='best.pt')
	model = model.autoshape()
	model.eval()
	model.cuda()
	#讀取影片路徑
	qt_env = QApplication(sys.argv)
	process = Qt()
	fileUrl = process.mv_Chooser()
	print(fileUrl)

	#開啟影片
	cap = cv2.VideoCapture(fileUrl)
	movie_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
	ret, frame = cap.read()
	#圖片處理
	i = 0
	
	while(ret):
		
		#將原先照片轉正
		start = time.time()
		frame = cv2.rotate(frame, cv2.cv2.ROTATE_90_CLOCKWISE)
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		mark_mat = frame.copy()
		results = model(gray, size=640)
		rect = results.xyxy[0][0].cpu().numpy()
		end = time.time()
		print(1/(end-start))
		cv2.rectangle(mark_mat, (rect[0], rect[1]), (rect[2], rect[3]), (255, 255, 255), 4, cv2.LINE_AA)
		cv2.imshow("results", mark_mat)
		ret, frame = cap.read()
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break



		
		