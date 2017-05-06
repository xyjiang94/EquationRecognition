import shelve
import imghdr# recognize img type
from os import listdir, getcwd, sep
from os.path import isfile, join
import re
from PIL import Image
import numpy as np
from scipy import misc
from skimage.morphology import binary_dilation,dilation,disk

sh = shelve.open("training",writeback=False)
imgFolderPath = getcwd() + sep + "annotated"
files = [f for f in listdir(imgFolderPath) if isfile(join(imgFolderPath, f)) and imghdr.what(join(imgFolderPath, f))=='png']
counter = 0

def input_wrapper(f):
	image = misc.imread(f)
	# image[image>50]=255
	# image[image<=50]=0
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

for e in files:
    h = re.match(".*eq\d*_(.*)_(\d*)_(\d*)_(\d*)_(\d*)\.png",e)
    if h is None:
        print "equation"
    else:
        print "snippet"
        counter+=1
        tmp = []
        symbol = h.group(1)
        if symbol == 'frac' or symbol == 'bar':
            symbol = '-'
        if symbol =='mul':
            symbol = 'x'
        img = input_wrapper(join(imgFolderPath, e))
        tmp.append(img)
        tmp.append(symbol)
        sh[str(counter)] = tmp
        #print tmp
sh.close()
print counter
