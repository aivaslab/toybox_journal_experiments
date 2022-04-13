import torch
import pickle
import csv
import cv2
import numpy as np


class DataLoaderGeneric(torch.utils.data.Dataset):
	"""
	This class
	"""
	def __init__(self, train=True, transform=None):
		self.train = train
		self.transform = transform
		
		if self.train:
			self.images_file = "../data_12/IN-12/train.pickle"
			self.labels_file = "../data_12/IN-12/train.csv"
		else:
			self.images_file = "../data_12/IN-12/test.pickle"
			self.labels_file = "../data_12/IN-12/test.csv"
		self.images = pickle.load(open(self.images_file, "rb"))
		self.labels = list(csv.DictReader(open(self.labels_file, "r")))
		
	def __len__(self):
		return len(self.images)

	def __getitem__(self, item):
		im = np.array(cv2.imdecode(self.images[item], 3))
		label = int(self.labels[item]["Class ID"])
		if self.transform is not None:
			im = self.transform(im)
		return item, im, label
