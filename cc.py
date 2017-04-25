import scipy
from scipy import ndimage
from scipy.misc import imread
import matplotlib.pyplot as plt
import numpy as np


fname='./equations/SKMBT_36317040717363_eq33.png'
blur_radius = 1.0
threshold = 50

img = imread(fname) # gray-scale image
print(img.shape)

# smooth the image (to remove small objects)
imgf = ndimage.gaussian_filter(img, blur_radius)
threshold = 50

# find connected components
labeled, nr_objects = ndimage.label(imgf > threshold)
print type(labeled)
imgs = ndimage.find_objects(labeled)
plt.imsave('out.png', labeled)
# print "Number of objects is %d " % nr_objects

s = []
for i in labeled:
    for j in i:
        if j not in s:
            s.append(j)
print len(s),s

print 'slices: ',len(imgs)
for i in range(len(imgs)):
    print imgs[i], type(imgs[i])
print imgs[0][0],type(imgs[0][0]),imgs[0][0].PySlice_GetIndices()

for k in range(0, nr_objects):
    label = k + 1
    bounding = imgs[k]
    x1 = (int)bounding[0][0]
    x2 = (int)bounding[0][1]
    y1 = (int)bounding[1][0]
    y2 = (int)bounding[1][1]
    len_x = x2 - x1
    len_y = y2 - y1

    background = np.zeros(len_x, len_y)

    for i  in range(x1, x2):
        for j in range(y1, y2):
            if labeled[i][j] == label:
                background[i - x1][j - y1] = 225

    plt.imsave('./tmp/'+str(k)+'.png', background)



#
# plt.imshow(labeled)
#
# plt.show()
