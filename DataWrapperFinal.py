import numpy as np
import random
from scipy import *
from skimage.transform import resize,warp,AffineTransform
import random

class MnistDigitsData(object):
	def __init__(self,imageList,labelsList):
		self.count =0
		self.labelList = labelsList
		self.imageList = imageList
	def image_deformation(self,image):
		random_shear_angl = np.random.random() * np.pi/6 - np.pi/12
		random_rot_angl = np.random.random() * np.pi/6 - np.pi/12 - random_shear_angl
		random_x_scale = np.random.random() * .4 + .8
		random_y_scale = np.random.random() * .4 + .8
		random_x_trans = np.random.random() * image.shape[0] / 4 - image.shape[0] / 8
		random_y_trans = np.random.random() * image.shape[1] / 4 - image.shape[1] / 8
		dx = image.shape[0]/2. \
				- random_x_scale * image.shape[0]/2 * np.cos(random_rot_angl)\
				+ random_y_scale * image.shape[1]/2 * np.sin(random_rot_angl + random_shear_angl)
		dy = image.shape[1]/2. \
				- random_x_scale * image.shape[0]/2 * np.sin(random_rot_angl)\
				- random_y_scale * image.shape[1]/2 * np.cos(random_rot_angl + random_shear_angl)
		trans_mat = AffineTransform(rotation=random_rot_angl,
									translation=(dx + random_x_trans,
												 dy + random_y_trans),
									shear = random_shear_angl,
									scale = (random_x_scale,random_y_scale))
		return warp(image,trans_mat.inverse,output_shape=image.shape)
	def get_valid(self,size = 500):
		images = np.zeros((size,32,32))
		labelsList = np.asarray(self.labelList)
		#print "The labelList shape is"
		#print labelsList.shape
		labels = labelsList[0:size,:]
		batch_x = np.zeros((size, 32, 32, 1))
		for i in range(500):
			images[i,:,:] = self.imageList[i]
			curr_patch = self.image_deformation(images[i, :, :])
			batch_x[i, :, :, 0] = curr_patch
		self.count = self.count+size
		images = np.reshape(images,(-1,32,32,1))
		return batch_x,labels
	def shuffle(self):
		pass
	def shuffleNum(self,batch_size):
		#Get batch_size of non-repeated random number
		wholeSize = len(self.labelList)
		shuffleList = random.sample(range(500, wholeSize), batch_size)
		#print shuffleList
		#print type(shuffleList[1])
		return shuffleList
	def next_batch(self,batch_size):
		images = np.zeros((batch_size,32,32))
		labelsList = np.asarray(self.labelList)
		#print labelsList.shape
		numList = self.shuffleNum(batch_size)
		#print numList
		#print type(numList[1])
		labels = labelsList[self.count:batch_size+self.count,:]
		#print labels
		#labels = []
		batch_x = np.zeros((batch_size, 32, 32, 1))
		for i in range(batch_size):
			k = numList[i]
			images[i,:,:] = self.imageList[k]
			curr_patch = self.image_deformation(images[i, :, :])
			batch_x[i, :, :, 0] = curr_patch
			labels[i,:] = labelsList[k]
		images = np.reshape(images,(-1,32,32,1))
		return batch_x,labels
