# add your imports here
from sys import argv
from glob import glob
from scipy import misc
import numpy as np
import random
from segmentation import *
import json
from MER_NN import SymbolRecognition
from MinimumSpanningTree import MinimumSpanningTree
from os.path import isfile, join, basename
from os import listdir, getcwd, sep
import tensorflow as tf
from partition import Partition
from classifyEq import Classify
"""
add whatever you think it's essential here
"""
# global variable
symMap = {}
with open('symbol_mapping.json', 'r') as opened:
	symMap = json.loads(opened.read())

class SymPred():
	def __init__(self, prediction, x1, y1, x2, y2):
		"""
		<x1,y1> <x2,y2> is the top-left and bottom-right coordinates for the bounding box
		(x1,y1)
			   .--------
			   |	   	|
			   |	   	|
			    --------.
			    		 (x2,y2)
		"""
		self.prediction = prediction
		self.x1 = x1
		self.y1 = y1
		self.x2 = x2
		self.y2 = y2

	def __str__(self):
		return self.prediction + '\t' + '\t'.join([
												str(self.x1),
												str(self.y1),
												str(self.x2),
												str(self.y2)])


class ImgPred():
	def __init__(self, image_name, sym_pred_list, latex='LATEX_REPR'):
		"""
		sym_pred_list is list of SymPred
		latex is the latex representation of the equation
		"""
		self.image_name = image_name
		self.latex = latex
		self.sym_pred_list = sym_pred_list

	def __str__(self):
		res = self.image_name + '\t' + \
		    str(len(self.sym_pred_list)) + '\t' + self.latex + '\n'
		for sym_pred in self.sym_pred_list:
			res += sym_pred[0]
			res += "\t"
			res += str(sym_pred[3])
			res += "\t"
			res += str(sym_pred[1])
			res += "\t"
			res += str(sym_pred[4])
			res += "\t"
			res += str(sym_pred[2])
			res += "\n"
		return res


def predict(image_path, sess, sr):
	"""
	Add your code here
	"""
	"""
	# Don't forget to store your prediction into ImgPred
	img_prediction = ImgPred(...)
	"""
	print image_path
	seg = Segmentation(image_path)
	d = seg.get_labels()
	mst = MinimumSpanningTree(d).get_mst()
	pa = Partition(mst, seg, sess, sr)
	l = pa.getList()
	c = Classify()
	result = c.classify(l)
	img_prediction = ImgPred(basename(image_path), l, result[1])
	return img_prediction


if __name__ == '__main__':
	model_path = join(getcwd(), "model", "model.ckpt")

	image_folder_path = argv[1]
	isWindows_flag = False
	if len(argv) == 3:
		isWindows_flag = True
	if isWindows_flag:
		image_paths = glob(image_folder_path + '\\*png')
	else:
		image_paths = glob(image_folder_path + '/*png')
	results = []
	with tf.Session() as sess:
		sr = SymbolRecognition(sess, model_path, trainflag=False)
		for image_path in image_paths:
			impred = predict(image_path, sess, sr)
			results.append(impred)

	with open('predictions.txt','w') as fout:
		for res in results:
			fout.write(str(res))
