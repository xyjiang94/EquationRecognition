from __future__ import division
import tensorflow as tf
import pickle
import numpy as np
import random
from skimage.morphology import binary_dilation,dilation,disk
from skimage.util import random_noise
from scipy import misc
from tensorflow.examples.tutorials.mnist import input_data
from DataWrapper import MnistDigitsData
from subprocess import call
from skimage.transform import resize,warp,AffineTransform
from scipy import ndimage

def input_wrapper(f):
	image = misc.imread(f)
	sx,sy = image.shape
	diff = np.abs(sx-sy)

	sx,sy = image.shape
	image = np.pad(image,((sx//8,sx//8),(sy//8,sy//8)),'constant')
	if sx > sy:
		image = np.pad(image,((0,0),(diff//2,diff//2)),'constant')
	else:
		image = np.pad(image,((diff//2,diff//2),(0,0)),'constant')
	
	image = dilation(image,disk(max(sx,sy)/32))
	image = misc.imresize(image,(32,32))
	if np.max(image) > 1:
		image = image/255.
	return image
class SymbolRecognition(object):
	def __init__(self,sess,model_path = None,symbols_path = 'symbols.txt',trainflag = True):
		self.sess = sess
		self.extra_size = 0
		self.random_brightness = .2
		self.random_br_l = 0.4
		self.random_br_h = 1.2
		self.random_contrast_l = 0.5
		self.random_contrast_h = 1.5
		self.trainflag = trainflag
		self.inference()
		self.saver = tf.train.Saver()
		symbol_list = []
		if model_path is not None:
			self.saver.restore(sess,model_path)
	
	def batch_norm_layer(self,inputs, decay = 0.9):
		is_training = self.trainflag
		epsilon = 1e-3
		scale = tf.Variable(tf.ones([inputs.get_shape()[-1]]))
		beta = tf.Variable(tf.zeros([inputs.get_shape()[-1]]))
		pop_mean = tf.Variable(tf.zeros([inputs.get_shape()[-1]]), trainable=False)
		pop_var = tf.Variable(tf.ones([inputs.get_shape()[-1]]), trainable=False)
		if is_training:
			batch_mean, batch_var = tf.nn.moments(inputs,[0,1,2])
			train_mean = tf.assign(pop_mean,
					pop_mean * decay + batch_mean * (1 - decay))
			train_var = tf.assign(pop_var,
					pop_var * decay + batch_var * (1 - decay))
			with tf.control_dependencies([train_mean, train_var]):
				return tf.nn.batch_normalization(inputs,
					batch_mean, batch_var, beta, scale, epsilon),batch_mean,batch_var
		else:
			return tf.nn.batch_normalization(inputs,
				pop_mean, pop_var, beta, scale, epsilon),pop_mean,pop_var

	def add_noise(self,curr_patch):
		if np.max(curr_patch) > 1:
			curr_patch /= 255.
		curr_patch = np.float32(curr_patch)
		curr_patch += (np.random.random() - .5) * self.random_brightness
		curr_patch *= np.random.random() * (self.random_br_h - self.random_br_l) + self.random_br_l
		curr_patch[curr_patch > 1] = 1
		curr_patch[curr_patch < 0] = 0
		#contrast adjust
		curr_patch = (curr_patch - np.mean(curr_patch)) \
			* (np.random.random() * (self.random_contrast_h - self.random_contrast_l) + self.random_contrast_l)\
			+ np.mean(curr_patch)
		curr_patch[curr_patch > 1] = 1
		curr_patch[curr_patch < 0] = 0
		return curr_patch
	
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
	
	def get_valid(self,dataset):
		data = dataset.get_valid()
		size = data[0].shape[0]
		print('valid size',size)
		target_num = data[1].shape[1]
		print(data[0].shape)
		images = data[0]
		print(images.shape)
		labels = data[1]
		batch_x = np.zeros((size,32,32,1))
		batch_y = np.zeros((size,target_num))
		for j in range(size):
			batch_y[j,:] = labels[j,:]
			curr_patch = self.image_deformation(images[j,:,:])
			batch_x[j,:,:,0] = curr_patch
		return batch_x,batch_y

	def next_batch(self,dataset,size = 128,target_num = 10):
		for i in range(20000):
			data = dataset.next_batch(size)
			images = data[0]
			labels = data[1]
			batch_x = np.zeros((size,32,32,1))
			batch_y = np.zeros((size,target_num))
			for j in range(size):
				batch_y[j,:] = labels[j,:]
				curr_patch = self.image_deformation(images[j,:,:])
				batch_x[j,:,:,0] = curr_patch
				
			yield batch_x,batch_y

	def weight_variable(self,shape):
		initial = tf.truncated_normal(shape, stddev=0.01)
		var = tf.Variable(initial)
		weight_decay = tf.multiply(tf.nn.l2_loss(var), 1e-5, name='weight_loss')
		tf.add_to_collection('losses', weight_decay)
		return var

	def bias_variable(self,shape):
		initial = tf.constant(0., shape=shape)
		return tf.Variable(initial)

	def conv2d(self,x, W,padding = 'SAME',stride = 1):
		return tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding=padding)	

	def max_pool_2x2(self,x,stride = 2):
		return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
								strides=[1, stride, stride, 1], padding='SAME')
	def avg_pool_global(self,x,ks):
		return tf.nn.avg_pool(x, ksize=[1, ks, ks, 1],
								strides=[1, 1, 1, 1], padding='VALID')

	def inference(self,target_num = 10):
		extra_size = self.extra_size
		if self.trainflag:
			self.x = tf.placeholder(tf.float32,[None,32,32,1])
			self.y_ = tf.placeholder(tf.float32,[None,target_num])
			padding = 'SAME'
		else:
			self.x = tf.placeholder(tf.float32,[1,None,None,1])
			padding = 'SAME'

		W_conv1 = self.weight_variable([3, 3, 1, 8])
		tmp_1,_,_ = self.batch_norm_layer(self.conv2d(self.x, W_conv1,padding=padding))
		h_conv1 = tf.nn.relu(tmp_1)

		W_conv2 = self.weight_variable([3, 3, 8, 8])	
		tmp_2,_,_ = self.batch_norm_layer(self.conv2d(h_conv1, W_conv2,padding=padding))
		h_conv2 = tf.nn.relu(tmp_2+self.x)

		#POOL
		W_skip2 = self.weight_variable([3,3,8,16])
		tmp_skip2,_,_ = self.batch_norm_layer(self.conv2d(h_conv2, W_skip2,padding=padding,stride = 2))
		h_skip2 = tf.nn.relu(tmp_skip2)

		W_conv3 = self.weight_variable([3, 3, 8, 16])	
		tmp_3,_,_ = self.batch_norm_layer(self.conv2d(h_conv2, W_conv3,padding=padding,stride = 2))
		h_conv3 = tf.nn.relu(tmp_3)

		W_conv4 = self.weight_variable([3, 3, 16, 16])
		tmp_4,_,_ = self.batch_norm_layer(self.conv2d(h_conv3, W_conv4,padding=padding))
		h_conv4 = tf.nn.relu(tmp_4+h_skip2)

		#POOL
		W_skip4 = self.weight_variable([1,1,16,32])
		tmp_skip4,_,_ = self.batch_norm_layer(self.conv2d(h_conv4, W_skip4,padding=padding,stride = 2))
		h_skip4 = tf.nn.relu(tmp_skip4)

		W_conv5 = self.weight_variable([3, 3, 16, 32])	
		tmp_5,_,_ = self.batch_norm_layer(self.conv2d(h_conv4, W_conv5,padding=padding,stride = 2))
		h_conv5 = tf.nn.relu(tmp_5)

		W_conv6 = self.weight_variable([3, 3, 32, 32])
		tmp_6,_,_ = self.batch_norm_layer(self.conv2d(h_conv5, W_conv6,padding=padding))
		h_conv6 = tf.nn.relu(tmp_6+h_skip4)

		h_pool6 = self.avg_pool_global(h_conv6,8)

		W_fc1 = self.weight_variable([1,1,32,64])
		b_fc1 = self.bias_variable([64])
		h_fc1 = self.conv2d(h_pool6, W_fc1,padding=padding) + b_fc1
		self.keep_prob = tf.placeholder(tf.float32)
		W_readout = self.weight_variable([1,1,64, target_num])
		b_readout = self.bias_variable([target_num])		
		readout = self.conv2d(h_fc1, W_readout,padding = padding) + b_readout
		self.h_fc1 = h_conv6
		if self.trainflag:
			readout = tf.reshape(readout,[-1,target_num])
			#self.y_conv=tf.nn.softmax(readout)
			self.y_conv = readout
			self.y_res = tf.argmax(self.y_conv,1)
			self.W_conv1 = W_conv1
		else:
			self.readout = readout
	def train(self, data, out_path = 'model_tt.ckpt',target_num = 10):
		#cross_entropy_mean = -tf.reduce_mean(self.y_ * tf.log(self.y_conv))
		cross_entropy_mean = tf.reduce_mean(
			tf.nn.softmax_cross_entropy_with_logits(labels=self.y_,logits=self.y_conv))
		tf.add_to_collection('losses', cross_entropy_mean)
		cross_entropy = tf.add_n(tf.get_collection('losses'), name='total_loss')
		self.l_rate = tf.placeholder(tf.float32)
		train_step = tf.train.AdamOptimizer(self.l_rate).minimize(cross_entropy)
		correct_prediction = tf.equal(tf.argmax(self.y_conv,1), tf.argmax(self.y_,1))
		accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
		sess = self.sess
		sess.run(tf.global_variables_initializer())
		#print(self.next_batch(256))
		data = MnistDigitsData()
		valid_x,valid_y = self.get_valid(data)

		learn_rate = 2e-3
		phist = .5
		for epic in range(1):
			i=0
			data.shuffle()
			for batch_x, batch_y in self.next_batch(data):			
				if i%100 == 0:
					train_accuracy,results,cem = sess.run([accuracy,self.y_res,cross_entropy_mean],
						feed_dict={	self.x:valid_x/255.0, self.y_: valid_y, 
									self.keep_prob: 1.0, self.l_rate: learn_rate})

					hit = np.zeros(target_num)
					precision = np.zeros(target_num)
					recall = np.zeros(target_num)
					for v_idx in range(1000):
						if valid_y[v_idx,results[v_idx]]:
							hit[results[v_idx]] += 1.0
						precision[results[v_idx]] += 1.0
						recall += valid_y[v_idx,:]
					print(hit/precision)
					print(hit/recall)

					print("step %d, training accuracy %g loss %g max %g %g lr %g"%(i, train_accuracy,cem,np.max(valid_x),np.max(batch_x),learn_rate))
					if np.abs(phist - train_accuracy) / phist < .1 :
						learn_rate /= 1.0
					if i % 2000 == 0:
						if learn_rate >= 1e-6:
							learn_rate /= 2.
					phist = train_accuracy
				train_step.run(feed_dict={self.x: batch_x/255.0, self.y_: batch_y, self.keep_prob: 1., self.l_rate: learn_rate	})
				i+=1
		print valid_x.shape
		for i in range(1000):
			misc.imsave('valid'+str(i)+'_'+str(np.argmax(valid_y[i,:]))+'.png',valid_x[i,:,:,0])
		save_path = self.saver.save(sess, out_path)
  		print("Model saved in file: %s" % save_path)

  	def p(self,image):
  		res = self.sess.run(self.readout,feed_dict={self.x: np.reshape(image,(1,32,32,1))/1.0, self.keep_prob: 1.0})
  		return np.argmax(res,axis=3).flatten()

if __name__ == '__main__':
	from sys import argv
	from scipy import misc
	from glob import glob
	if argv[1] == 'train':
		with tf.Session() as sess:
			sr = SymbolRecognition(sess)
			sr.train(argv[3],argv[2])
	else:
		with tf.Session() as sess:
			print 'model_path,',argv[2]
			sr = SymbolRecognition(sess,argv[2],trainflag = False)
			ann_dict = dict()
			with open('annotation.txt','r') as fin:
				for line in fin:
					if '\t' in line:
						ss = line.split('\t')
						ann_dict[ss[0]] = int(ss[1])

			acc = 0
			fid = 0
			call(['rm','trans/*'])
			call(['rm','err/*'])
			for f in glob('examples/*png'):
				image = input_wrapper(f)
				p = sr.p(image)
				p = p[0]
				misc.imsave('trans/'+str(p)+'_'+f[9:],np.reshape(image,(32,32)))
				if p == ann_dict[f[9:]]:
					acc += 1
				else:
					misc.imsave('err/'+str(p)+'_'+f[9:],np.reshape(image,(32,32)))
				fid +=1.
			print(acc,fid,acc/fid)