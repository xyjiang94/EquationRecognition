import scipy
from scipy import ndimage
from scipy import misc
import numpy as np
import sys
import datetime

class Segmentation(object):

    # Parameters
    # ----------
    # img_path : string
    def __init__(self,path):
        blur_radius = 1.0
        threshold = 50

        img = misc.imread(path)

        # smooth the image (to remove small objects)
        imgf = ndimage.gaussian_filter(img, blur_radius)
        threshold = 50
        # find connected components
        self.img, self.count = ndimage.label(imgf > threshold)
        self.labels = self.calculate_labels()


    # Return
    # ----------
    # labels : dictionary, key = label, value = list of bounding in order y1, y2, x1, x2
    def get_labels(self):
        return self.labels


    def calculate_labels(self):
        labels = {}
        for label in range(1, self.count + 1):
            labels[label] = [sys.maxint,-1,sys.maxint,-1]

        img = self.img
        for row in range(img.shape[0]):
            for col in range(img.shape[1]):
                value = img[row][col]
                if value != 0:
                    bounding = labels[value]
                    if bounding[0] > row:
                        bounding[0] = row
                    if bounding[2] > col:
                        bounding[2] = col
                    if bounding[1] < row:
                        bounding[1] = row
                    if bounding[3] < col:
                        bounding[3] = col
        return labels


    # Parameters
    # ----------
    # label : int
    #     Label of the stroke to be found
    #
    # Returns
    # -------
    # np.ndarray
    # the stroke image in format of np array
    def get_stroke(self,label):
        l = self.labels[label]

        stroke = np.copy(self.img[l[0]:l[1],l[2]:l[3]])

        for data in np.nditer(stroke, op_flags=['readwrite']):
            if data != label:
                data[...] = 0
            else:
                data[...] = 225

        return stroke


    # Parameters
    # ----------
    # l_labels : list
    #     A list of stroke labels representing the strokes to be combined
    #
    # Returns
    # -------
    # np.ndarray
    # the combined stroke image in format of np array
    def get_combined_strokes(self,l_labels):
        bounding = self.get_combined_bounding(l_labels)
        stroke = np.copy(self.img[bounding[0]:bounding[1],bounding[2]:bounding[3]])

        for data in np.nditer(stroke, op_flags=['readwrite']):
            if data in l_labels:
                data[...] = 225
            else:
                data[...] = 0

        return stroke


    # Parameters
    # ----------
    # l_labels : list
    #     A list of stroke labels representing the strokes to be combined
    #
    # Returns
    # -------
    # list
    # the list of bounding of combined strokes
    def get_combined_bounding(self,l_labels):
        l = [sys.maxint,-1,sys.maxint,-1]
        for label in l_labels:
            bounding = self.labels[label]
            if bounding[0] < l[0]:
                l[0] = bounding[0]
            if bounding[1] > l[1]:
                l[1] = bounding[1]
            if bounding[2] < l[2]:
                l[2] = bounding[2]
            if bounding[3] > l[3]:
                l[3] = bounding[3]
        return l


if __name__ == '__main__':
    fname='./equations/SKMBT_36317040717260_eq6.png'
    seg = Segmentation(fname)

    print seg.labels
    # for label in seg.labels.keys():
    #     print label
    #     stroke = seg.get_stroke(label)
    #     scipy.misc.imsave('./tmp/'+ str(label)+'.png', stroke)
    #
    # combined = seg.get_combined_strokes([1,2])
    # scipy.misc.imsave('./tmp/combined.png', combined)
