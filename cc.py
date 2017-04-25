import scipy
from scipy import ndimage
from scipy.misc import imread
import matplotlib.pyplot as plt

fname='1.png'
blur_radius = 1.0
threshold = 50

img = imread(fname) # gray-scale image
print(img.shape)

# smooth the image (to remove small objects)
imgf = ndimage.gaussian_filter(img, blur_radius)
threshold = 50

# find connected components
labeled, nr_objects = ndimage.label(imgf > threshold)
print "Number of objects is %d " % nr_objects

plt.imsave('out.png', labeled)
plt.imshow(labeled)

plt.show()
