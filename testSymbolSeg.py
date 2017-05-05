import shelve
import imghdr# recognize img type
from os import listdir, getcwd, sep
from os.path import isfile, join
import re
from PIL import Image
import scipy
import numpy as np
from scipy import ndimage
from scipy import misc
from skimage.morphology import binary_dilation,dilation,disk
np.set_printoptions(threshold=np.nan)




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
	print image
	return image

fname='/Users/zhengyjo/Desktop/EquationRecognition/compare/original/SKMBT_36317040717260_eq23_+_64_90_923_961.png'
handle = input_wrapper(fname)
scipy.misc.imsave('/Users/zhengyjo/Desktop/EquationRecognition/compare/original/SKMBT_36317040717260_eq23_+_64_90_923_961_Wrapper.png', handle)

