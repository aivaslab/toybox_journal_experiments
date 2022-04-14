import torch
import pickle
import csv
import cv2
import numpy as np


class DataLoaderGeneric(torch.utils.data.Dataset):
	"""
	This class
	"""
	def __init__(self, root, train=True, transform=None):
		self.train = train
		self.transform = transform
		self.root = root
		
		if self.train:
			self.images_file = self.root + "train.pickle"
			self.labels_file = self.root + "train.csv"
		else:
			self.images_file = self.root + "test.pickle"
			self.labels_file = self.root + "test.csv"
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
