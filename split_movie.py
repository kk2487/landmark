import os
import sys
import cv2 
import numpy as np
from tqdm import tqdm
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
import PyQt5.QtGui as QtGui
from PyQt5.QtCore import *
from pathlib import Path  
import shutil
class Qt(QWidget):
	def mv_Chooser(self):    
		opt = QFileDialog.Options()
		opt |= QFileDialog.DontUseNativeDialog
		fileUrl = QFileDialog.getOpenFileName(self,"Input Video", "C:/Users/hongze/Desktop/","Mp4 (*.mp4)", options=opt)
	
		return fileUrl[0]

if __name__ == '__main__':

	#讀取影片路徑
	qt_env = QApplication(sys.argv)
	process = Qt()
	fileUrl = process.mv_Chooser()
	print(fileUrl)

	#建立資料夾用以儲存Data
	filename = Path(fileUrl).stem

	if(not os.path.exists('./dataset')):
		os.makedirs('./dataset')
		os.makedirs('./dataset/train')
		os.makedirs('./dataset/validation')

	#開啟影片
	cap = cv2.VideoCapture(fileUrl)
	movie_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
	
	#圖片處理
	i = 0
	n = 0
	for i in tqdm(range(movie_length)):
		
		if(i%80 == 0):
			ret, frame = cap.read()

			#將原先照片轉正
			frame = cv2.rotate(frame, cv2.cv2.ROTATE_90_CLOCKWISE)
			#frame = cv2.rotate(frame, cv2.cv2.ROTATE_90_COUNTERCLOCKWISE)
			gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
			if(n%5 ==0):
				savefile = "./dataset/validation/pic_"+filename +"_" +str(i) + ".jpg"
				cv2.imwrite(savefile, gray)
			else:
				savefile = "./dataset/train/pic_"+filename +"_" +str(i) + ".jpg"
				cv2.imwrite(savefile, gray)
			n = n+1
		