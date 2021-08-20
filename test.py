import os
import sys
import time
import warnings
import numpy as np
from PIL import Image
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
import PyQt5.QtGui as QtGui
from PyQt5.QtCore import *
import torch
import glob
import torch.nn as nn
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torchvision
import pathlib
import cv2
import CNN as cnn

warnings.filterwarnings('ignore')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_path = './best_checkpoint.model'

class Qt(QWidget):
    def mv_Chooser(self):    
        opt = QFileDialog.Options()
        opt |= QFileDialog.DontUseNativeDialog
        fileUrl = QFileDialog.getOpenFileName(self,"Input Video", "./","Mp4 (*.mp4)", options=opt)
	
        return fileUrl[0]

transformer=transforms.Compose([
	transforms.Resize((256,256)),
	transforms.ToTensor(),
	transforms.Normalize(mean=[0.2], std=[0.58])
])

if __name__ == '__main__':

	qt_env = QApplication(sys.argv)
	process = Qt()
	fileUrl = process.mv_Chooser()
	print(fileUrl)
	if(fileUrl == ""):
		print("Without input file!!")
		sys.exit(0)

	checkpoint=torch.load(model_path)
	model=cnn.PFLDInference().to(device)
	model.load_state_dict(checkpoint)
	model.eval()

	cap = cv2.VideoCapture(fileUrl)
	ret, frame = cap.read()

	height, width = frame.shape[:2]

	videoWriter = cv2.VideoWriter("./result.avi",cv2.VideoWriter_fourcc('X','V','I','D'),30,(width,height))
	while(ret):

		start = time.time()

		ret, frame = cap.read()
		if(not ret):
			break
		draw = frame.copy()

		frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
		#frame = cv2.resize(frame,(112,112))
		img=Image.fromarray(np.uint8(frame))
		img = transformer(img).unsqueeze(0).to(device)
		
		out = model(img)
		out = out.cpu().detach().numpy()
		out = out[0]
		print(out)

		for i in range(0,28,2):
			cv2.circle(draw,(int(out[i] * width),int(out[i+1] * height)), 5, (255), -1)
		end = time.time()
		
		print(1/(end-start))
		cv2.imshow("frame", draw)
		videoWriter.write(draw)
		if cv2.waitKey(1) == 27:
			cap.release()
			videoWriter.release()
			cv2.destroyAllWindows()
			break

	cap.release()
	videoWriter.release()
	cv2.destroyAllWindows()