import torch
import pickle
import csv
import cv2
import numpy as np


class DataLoaderGeneric(torch.utils.data.Dataset):
	"""
	This class
	"""
	def __init__(self, root, train=True, transform=None, fraction=1.0):
		self.train = train
		self.transform = transform
		self.root = root
		self.fraction = fraction
		
		if self.train:
			self.images_file = self.root + "train.pickle"
			self.labels_file = self.root + "train.csv"
		else:
			self.images_file = self.root + "test.pickle"
			self.labels_file = self.root + "test.csv"
		self.images = pickle.load(open(self.images_file, "rb"))
		self.labels = list(csv.DictReader(open(self.labels_file, "r")))
		if self.train:
			if self.fraction < 1.0:
				len_all_images = len(self.images)
				len_train_images = int(len_all_images * self.fraction)
				rng = np.random.default_rng(0)
				self.selected_indices = rng.choice(len_all_images, len_train_images, replace=False)
			else:
				self.selected_indices = np.arange(len(self.images))
				
	def __len__(self):
		if self.train:
			return len(self.selected_indices)
		else:
			return len(self.images)

	def __getitem__(self, index):
		if self.train:
			item = self.selected_indices[index]
		else:
			item = index
		im = np.array(cv2.imdecode(self.images[item], 3))
		label = int(self.labels[item]["Class ID"])
		if self.transform is not None:
			im = self.transform(im)
		return index, im, label
