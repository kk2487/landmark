import os
import sys
import numpy as np
import torch
import glob
import torch.nn as nn
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.autograd import Variable
import torchvision
import pathlib
from tqdm import tqdm

from dataloader import MYDataset
import cv2
num_epochs = 500
batch_size = 8
img_path='./dataset/img/'
"""
class ConvNet(nn.Module):
	def __init__(self,num_classes=30):
		super(ConvNet,self).__init__()
		
		#Input shape= (3,256,256)
		
		self.conv1=nn.Conv2d(in_channels=1,out_channels=32,kernel_size=3,stride=1,padding=1)
		#Shape= (32,256,256)
		self.relu1=nn.ReLU()
		#Shape= (32,256,256)
		self.pool1=nn.MaxPool2d(kernel_size=2)
		#Shape= (32,128,128)
		
		
		self.conv2=nn.Conv2d(in_channels=32,out_channels=64,kernel_size=3,stride=1,padding=1)
		#Shape= (64,128,128)
		self.relu2=nn.ReLU()
		#Shape= (64,128,128)
		self.pool2=nn.MaxPool2d(kernel_size=2)
		#Shape= (64,64,64)
		
		
		self.conv3=nn.Conv2d(in_channels=64,out_channels=128,kernel_size=3,stride=1,padding=1)
		#Shape= (128,64,64)
		self.relu3=nn.ReLU()
		#Shape= (128,64,64)
		self.pool3=nn.MaxPool2d(kernel_size=2)
		#Shape= (128,32,32)        
		
		self.conv4=nn.Conv2d(in_channels=128,out_channels=64,kernel_size=3,stride=1,padding=1)
		#Shape= (64,32,32)
		self.relu4=nn.ReLU()
		#Shape= (64,32,32)
		self.pool4=nn.MaxPool2d(kernel_size=2)
		#Shape= (64,16,16)   
  
		self.conv5=nn.Conv2d(in_channels=64,out_channels=32,kernel_size=3,stride=1,padding=1)
		#Shape= (32,16,16)
		self.relu5=nn.ReLU()
		#Shape= (32,16,16)
		self.pool5=nn.MaxPool2d(kernel_size=2)
		#Shape= (32,8,8) 
		
		self.fc1=nn.Linear(in_features=8 * 8 * 32,out_features=1024)
		self.relu6=nn.ReLU()
		self.dropout1=nn.Dropout(p=0.6)

		self.fc2=nn.Linear(in_features=1024,out_features=512)
		self.relu7=nn.ReLU()
		self.dropout2=nn.Dropout(p=0.6)

		self.fc3=nn.Linear(in_features=512,out_features=num_classes)
		
		#Feed forwad function
		
	def forward(self,input):

		output=self.conv1(input)
		output=self.relu1(output)
		output=self.pool1(output)
		  
			
		output=self.conv2(output)
		output=self.relu2(output)
		output=self.pool2(output)
		
		output=self.conv3(output)
		output=self.relu3(output)
		output=self.pool3(output)
			
		output=self.conv4(output)
		output=self.relu4(output)
		output=self.pool4(output)    
		
		output=self.conv5(output)
		output=self.relu5(output)
		output=self.pool5(output)
		
		output=output.view(-1, 8 * 8 * 32)        
		
		output=self.fc1(output)
		output=self.relu6(output)
		output=self.dropout1(output)
		
		output=self.fc2(output)
		output=self.relu7(output)
		output=self.dropout2(output)
		
		output=self.fc3(output)
			
		return output
"""
def conv_bn(inp, oup, kernel, stride, padding=1):
    return nn.Sequential(
        nn.Conv2d(inp, oup, kernel, stride, padding, bias=False),
        nn.BatchNorm2d(oup), nn.ReLU(inplace=True))


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, use_res_connect, expand_ratio=6):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        self.use_res_connect = use_res_connect

        self.conv = nn.Sequential(
            nn.Conv2d(inp, inp * expand_ratio, 1, 1, 0, bias=False),
            nn.BatchNorm2d(inp * expand_ratio),
            nn.ReLU(inplace=True),
            nn.Conv2d(inp * expand_ratio,
                      inp * expand_ratio,
                      3,
                      stride,
                      1,
                      groups=inp * expand_ratio,
                      bias=False),
            nn.BatchNorm2d(inp * expand_ratio),
            nn.ReLU(inplace=True),
            nn.Conv2d(inp * expand_ratio, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class PFLDInference(nn.Module):
    def __init__(self):
        super(PFLDInference, self).__init__()

        self.conv1 = nn.Conv2d(1,
                               64,
                               kernel_size=3,
                               stride=2,
                               padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(64,
                               64,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        self.conv3_1 = InvertedResidual(64, 64, 2, False, 2)

        self.block3_2 = InvertedResidual(64, 64, 1, True, 2)
        self.block3_3 = InvertedResidual(64, 64, 1, True, 2)
        self.block3_4 = InvertedResidual(64, 64, 1, True, 2)
        self.block3_5 = InvertedResidual(64, 64, 1, True, 2)

        self.conv4_1 = InvertedResidual(64, 128, 2, False, 2)

        self.conv5_1 = InvertedResidual(128, 128, 1, False, 4)
        self.block5_2 = InvertedResidual(128, 128, 1, True, 4)
        self.block5_3 = InvertedResidual(128, 128, 1, True, 4)
        self.block5_4 = InvertedResidual(128, 128, 1, True, 4)
        self.block5_5 = InvertedResidual(128, 128, 1, True, 4)
        self.block5_6 = InvertedResidual(128, 128, 1, True, 4)

        self.conv6_1 = InvertedResidual(128, 16, 1, False, 2)  # [16, 32, 32]

        self.conv7 = conv_bn(16, 32, 3, 2)  # [32, 7, 7]
        self.conv8 = nn.Conv2d(32, 128, 7, 1, 0)  # [128, 1, 1]
        self.bn8 = nn.BatchNorm2d(128)

        self.avg_pool1 = nn.AvgPool2d(32)
        self.avg_pool2 = nn.AvgPool2d(16)
        self.fc = nn.Linear(12848, 30)

    def forward(self, x):  # x: 3, 112, 112
        x = self.relu(self.bn1(self.conv1(x)))  # [64, 128, 128]
        x = self.relu(self.bn2(self.conv2(x)))  # [64, 128, 128]
        x = self.conv3_1(x)
        x = self.block3_2(x)
        x = self.block3_3(x)
        x = self.block3_4(x)
        out1 = self.block3_5(x)

        x = self.conv4_1(out1)
        x = self.conv5_1(x)
        x = self.block5_2(x)
        x = self.block5_3(x)
        x = self.block5_4(x)
        x = self.block5_5(x)
        x = self.block5_6(x)
        x = self.conv6_1(x)
        #print(x.shape)
        x1 = self.avg_pool1(x)
        x1 = x1.view(x1.size(0), -1)

        x = self.conv7(x)
        #print(x.shape)
        x2 = self.avg_pool2(x)
        x2 = x2.view(x2.size(0), -1)

        x3 = self.relu(self.conv8(x))
        #print(x.shape)
        x3 = x3.view(x3.size(0), -1)

        multi_scale = torch.cat([x1, x2, x3], 1)
        #print(multi_scale.shape)
        landmarks = self.fc(multi_scale)

        return landmarks


def load_model():

	checkpoint=torch.load(user_set.model_path)
	#model=ConvNet(num_classes=30)
	model=PFLDInference()
	model.load_state_dict(checkpoint)
	return model

def write_classes(file_path, classes):

	fp = open(file_path, "w")
	for i in range(len(classes)):
		fp.write(str(classes[i]))
		if (i == len(classes)-1):
			break
		fp.write(',')
	fp.close()

def read_classes(file_path):

	fp = open(file_path, "r")
	classes = fp.readline()
	classes = classes. split(",")
	fp.close()

	return classes


if __name__ == '__main__':

	#if(sys.argv[1] == '--train'):

	device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	print('-------------------------------------')
	print('Training Device :')
	print(device)
	print(torch.cuda.get_device_name(0))
	print('Memory Usage:')
	print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
	print('Cached:   ', round(torch.cuda.memory_reserved(0)/1024**3,1), 'GB')
	print('-------------------------------------')

	#Dataloader

	data = MYDataset()
	print(len(data))
	model=PFLDInference().to(device)

	#Optmizer and loss function
	optimizer=Adam(model.parameters(),lr=0.001,weight_decay=0.0001)
	criterion  = nn.MSELoss()
	
	train_dataloader = DataLoader(dataset=data, batch_size=batch_size, shuffle=True)
	
	#Model training and saving best model

	best_loss=1000.0

	train_count=len(glob.glob(img_path+'/**/*.jpg'))

	for epoch in range(num_epochs):

		#Evaluation and training on training dataset
		model.train()

		train_loss=0.0

		for (images,labels) in tqdm(train_dataloader):

			print(images[0])
			out_mat = images[0].cpu().detach().numpy()
			mat = out_mat.reshape(256,256)
			print(labels[0])
			out = labels[0].cpu().detach().numpy()
			print(out)
			for i in range(0,28,2):
				cv2.circle(mat,(int(out[i] * 256),int(out[i+1] * 256)), 5, (255), -1)
			cv2.imshow("frame", mat)

			if cv2.waitKey(100) == 27:
				cv2.destroyAllWindows()
			break