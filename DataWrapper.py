import numpy as np
import random
from tensorflow.examples.tutorials.mnist import input_data
from scipy import misc

class MnistDigitsData(object):
	def __init__(self):
		self.mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
	def get_valid(self,size = 1000):
		data = self.mnist.train.next_batch(size)
		images = np.zeros((size,32,32))
		labels = data[1]
		for i in range(1000):
			images[i,:,:] = misc.imresize(np.reshape(data[0][i],(28,28)),(32,32))
		return images,labels
	def shuffle(self):
		pass
	def next_batch(self,batch_size):
		data = self.mnist.train.next_batch(batch_size)
		images = np.zeros((batch_size,32,32))
		labels = data[1]
		for i in range(batch_size):
			images[i,:,:] = misc.imresize(np.reshape(data[0][i],(28,28)),(32,32))
		return images,labels
