import numpy as np
import random
from scipy import misc
from readDB import *
import random

class MnistDigitsData(object):
	def __init__(self):
		self.count =0
		self.labelList = labelsList
		self.imageList = imageList
	def get_valid(self,size = 1000):
		images = np.zeros((size,32,32))
		labelsList = np.asarray(self.labelList)
		#print "The labelList shape is"
		#print labelsList.shape
		labels = labelsList[0:size,:]
		for i in range(1000):
			images[i,:,:] = imageList[i]
		self.count = self.count+size
		return images,labels
	def shuffle(self):
		pass
	def shuffleNum(self,batch_size):
		#Get batch_size of non-repeated random number
		wholeSize = len(labelsList)
		shuffleList = random.sample(range(1000, wholeSize), batch_size)
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
		#labels = []
		for i in range(batch_size):
			k = numList[i]
			images[i,:,:] = imageList[k]
			labels[i,:] = labelsList[k]
		#self.count = self.count + batch_size
		return images,labels