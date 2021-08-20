import os
import torch
from torchvision.transforms import transforms
from torch.utils.data import Dataset
from PIL import Image
import numpy as np

transformer=transforms.Compose([
	#transforms.RandomRotation(90),
	transforms.Resize((256,256)),
	transforms.ToTensor(),
	transforms.Normalize(mean=[0.2], std=[0.58])
])

#transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
img_path='./dataset/img/'
label_path='./dataset/label/label.txt'

def read_label(path):
	label = []
	txt_file = open(path, "r")
	xtlines = txt_file.readlines()
	for item in xtlines:
		single_label = []
		item = item.split(" ")
		for i in range(1,31):
			if i == 30:
				item[i] = item[i][:-1]
			item[i] = float(item[i])	
			single_label.append(item[i])
		label.append(single_label)
	label = np.array(label)
	#print(len(label))
	label = torch.from_numpy(label)

	return label

def read_image(path):
	image = []

	for item in os.listdir(path):
		with open(img_path+item, 'rb') as f:
			with Image.open(f) as img:
				image.append(transformer(img.convert('L')))

	return image

class MYDataset(Dataset):
	def __init__(self):
		
		self.images = read_image(img_path)
		self.label = read_label(label_path)
		
	def __len__(self):
		return len(self.images)

	def __getitem__(self, index):
		x = self.images[index].float()
		y = self.label[index].float()
		return x, y