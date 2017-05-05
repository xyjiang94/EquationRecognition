# TODO
# input mst
# ----------
# dictionary :  key = vertex, value = a list of tuples (tuple: [connected vertex, weight])
# output lst
# ----------
# lists :  list of format [symbol, x1, y1, x2, y2]
from segmentation import Segmentation
from MinimumSpanningTree import MinimumSpanningTree
from collections import deque
import imghdr  # recognize img type
from os import listdir, getcwd, sep
from os.path import isfile, join
from subprocess import call
import re
import json
import tensorflow as tf
from scipy import misc
import numpy as np
from MER_NN import SymbolRecognition
from skimage.morphology import binary_dilation,dilation,disk


symMap = {}
with open('symbol_mapping.json', 'r') as opened:
    symMap = json.loads(opened.read())
print symMap

model_path = join(getcwd(), "model", "model.ckpt")

class Partition(object):
    def __init__(self,mst,seg):
        self.mst = mst
        self.seg = seg
        self.lst = []
        self.generateList()

    def getList(self):
        return self.lst

    def generateList(self):
        generated = []
        lowProb = []
        with tf.Session() as sess:
            sr = SymbolRecognition(sess, model_path, trainflag=False)
            visited = set([])
            queue = deque([1])
            while len(queue)>0:
                v = queue.popleft()
                visited.add(v)
                image = seg.get_combined_strokes([v])
                bb = seg.get_combined_bounding([v])
                image = self.input_wrapper_arr(image)
                test = sr.pr(image)
                p = sr.p(image)
                probability = sess.run(tf.nn.softmax(test)[0][0][0][p[0]])
                p = symMap[str(p[0])]
                print probability,p
                if probability>0.8 :
                    self.lst.append([p,bb[0],bb[1],bb[2],bb[3],[v]])
                    generated.append(v)
                    if p=="-":
                        if len(self.lst)>1:
                            if self.lst[-2][0]=="-":
                                if abs(bb[2]-self.lst[-2][3])<15 and abs(bb[3]-self.lst[-2][4])<15:
                                    l = [generated[-2],v]
                                    bb = seg.get_combined_bounding(l)
                                    w = self.lst[-2][-1]
                                    self.lst.pop()
                                    self.lst.pop()
                                    self.lst.append(["=",bb[0],bb[1],bb[2],bb[3],[w,v]])
                            else:
                                x = (self.lst[-2][3]+self.lst[-2][4])/2
                                if x>bb[2] and x<bb[3]:
                                    self.lst[-1][0] = "frac"
                    elif len(self.lst)>1 and self.lst[-2][0]=="-":
                        x = (bb[2]+bb[3])/2
                        if x>self.lst[-2][3] and x<self.lst[-2][4]:
                            self.lst[-2][0] = "frac"
                    elif p=="x" and len(self.lst)>1 and self.lst[-2][0] in ["a","b","c","d","frac"]:
                        self.lst[-1][0]="mul"
                else:
                    lowProb.append(v)
                    if len(lowProb)>1 and len(self.lst)>0 and self.lst[-1][0]=="-":
                        l = [lowProb[-2],self.lst[-1][-1][0],v]
                        image = seg.get_combined_strokes(l)
                        bb = seg.get_combined_bounding(l)
                        image = self.input_wrapper_arr(image)
                        test = sr.pr(image)
                        p = sr.p(image)
                        probability = sess.run(tf.nn.softmax(test)[0][0][0][p[0]])
                        p = symMap[str(p[0])]
                        print probability,l,p
                        if probability>0.8 :
                            self.lst.pop()
                            self.lst.append([p,bb[0],bb[1],bb[2],bb[3],l])
                            lowProb.pop()
                            lowProb.pop()


                for w in self.mst[v]:
                    if w[0] in visited:
                        continue
                    queue.append(w[0])

            for e in lowProb:
                if e in generated:
                    continue
                queue = deque([e])
                conn = []
                while len(queue)>0:
                    v = queue.popleft()
                    conn.append(v)
                    generated.append(v)
                    for w in self.mst[v]:
                        if w[0] in generated:
                            continue
                        queue.append(w[0])
                image = seg.get_combined_strokes(conn)
                bb = seg.get_combined_bounding(conn)
                image = self.input_wrapper_arr(image)
                test = sr.pr(image)
                p = sr.p(image)
                probability = sess.run(tf.nn.softmax(test)[0][0][0][p[0]])
                p = symMap[str(p[0])]
                print probability,conn,p
                if probability>0.5 :
                    self.lst.append([p,bb[0],bb[1],bb[2],bb[3],conn])


    def input_wrapper_arr(self,image):
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

fname='./equations/SKMBT_36317040717260_eq16.png'
seg = Segmentation(fname)
d = seg.get_labels()
mst = MinimumSpanningTree(d).get_mst()
print mst
pa = Partition(mst,seg)
print pa.getList()
