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
from FacePose_pytorch.dectect import AntiSpoofPredict

class Qt(QWidget):
	def mv_Chooser(self):    
		opt = QFileDialog.Options()
		opt |= QFileDialog.DontUseNativeDialog
		fileUrl = QFileDialog.getOpenFileName(self,"Input Video", "C:/Users/hongze/Desktop/","Mp4 (*.mp4)", options=opt)
	
		return fileUrl[0]

face_x1 = 0
face_y1 = 0
face_x2 = 0
face_y2 = 0

if __name__ == '__main__':

	#讀取影片路徑
	fp = open("data.txt", "r")
	lines = fp.read()
	print(lines)
	fp.close()

	qt_env = QApplication(sys.argv)
	process = Qt()
	fileUrl = process.mv_Chooser()
	print(fileUrl)
	if(fileUrl == ""):
		print("Without input file!!")
		sys.exit(0)

	#建立資料夾用以儲存Data
	video_filename = Path(fileUrl).stem

	fp = open("data.txt", "r")
	found = False
	print(fp)
	for line in fp:
		if video_filename in line:
			found = True
			break

	fp.close()
	if(found):
		print("Already split!")
		sys.exit(0)

	file = "dataset"
	if not os.path.exists(file):
		os.makedirs(file)
		os.makedirs(file+'/training')
		os.makedirs(file+'/validation')

	#開啟影片
	cap = cv2.VideoCapture(fileUrl)
	movie_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
	
	#圖片處理

	face_model = AntiSpoofPredict(0)
	

	for i in tqdm(range(movie_length)):
		ret, frame = cap.read()

		#將原先照片轉正
		frame = cv2.rotate(frame, cv2.cv2.ROTATE_90_CLOCKWISE)
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

		image_bbox = face_model.get_bbox(frame)
		face_x1 = image_bbox[0]
		face_y1 = image_bbox[1]
		face_x2 = image_bbox[0] + image_bbox[2]
		face_y2 = image_bbox[1] + image_bbox[3]
		
		
		#cv2.rectangle(frame, (face_x1, face_x1), (face_x2, face_y2), (255, 255, 255), 4, cv2.LINE_AA)
		cropped = gray[int(face_y1):int(face_y2), int(face_x1):int(face_x2)]
		cropped = cv2.resize(cropped, (112,112))
		if(image_bbox[2] > 50 and image_bbox[3] > 50):

			if(i%100 == 0):
				path = './'+file+'/validation/'
				validation_num = len([name for name in os.listdir(path) if os.path.isfile(os.path.join(path, name))])
				savefile = file + "/validation/pic_"+str(validation_num+1) + ".jpg"
				cv2.imwrite(savefile, cropped)

			elif(i%10 == 0):
				path = './'+file+'/training/'
				training_num = len([name for name in os.listdir(path) if os.path.isfile(os.path.join(path, name))])
				savefile = file + "/training/pic_"+str(training_num+1) + ".jpg"
				cv2.imwrite(savefile, cropped)


		cv2.imshow("face", cropped)

		if cv2.waitKey(1) & 0xFF == ord('q'):
			cap.release()
			cv2.destroyAllWindows()
			break
	fp = open("data.txt", "a")
	fp.write(video_filename+"\n")
	fp.close()
		
		