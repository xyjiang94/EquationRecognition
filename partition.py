# TODO
# input mst
# ----------
# dictionary :  key = vertex, value = a list of tuples (tuple: [connected vertex, weight])
# output lst
# ----------
# lists :  list of format [symbol, x1, y1, x2, y2]
from segmentation import Segmentation
from MinimumSpanningTree import MinimumSpanningTree
from collections import deque, defaultdict
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
import scipy

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
        self.count = defaultdict(lambda:0)

    def getList(self):
        return self.lst

    def calculateCount(self):
        for e in self.lst:
            self.count[e[0]]+=1

    def getCount(self):
        return self.count

    def generateList(self):
        generated = []
        dots = []
        with tf.Session() as sess:
            sr = SymbolRecognition(sess, model_path, trainflag=False)
            visited = set([])
            queue = deque([1])
            while len(queue)>0:
                v = queue.popleft()
                visited.add(v)
                image = self.seg.get_combined_strokes([v])
                bb = self.seg.get_combined_bounding([v])
                image = self.input_wrapper_arr(image)
                test = sr.pr(image)
                p = sr.p(image)
                probability = sess.run(tf.nn.softmax(test)[0][0][0][p[0]])
                p = symMap[str(p[0])]
                print probability,p
                if probability>0.5 :
                    self.lst.append([p,bb[0],bb[1],bb[2],bb[3],[v]])
                    generated.append(v)
                    if p=="-":
                        print p,bb
                        # if len(self.lst)>1:
                        #     if self.lst[-2][0]=="-":
                        #         if abs(bb[2]-self.lst[-2][3])<15 and abs(bb[3]-self.lst[-2][4])<15:
                        #             l = [generated[-2],v]
                        #             bb = self.seg.get_combined_bounding(l)
                        #             w = self.lst[-2][-1]
                        #             self.lst.pop()
                        #             self.lst.pop()
                        #             self.lst.append(["=",bb[0],bb[1],bb[2],bb[3],[w,v]])
                        #     else:
                        #         x = (self.lst[-2][3]+self.lst[-2][4])/2
                        #         y = (self.lst[-2][1]+self.lst[-2][2])/2
                        #         if x>bb[2] and x<bb[3]:
                        #             if y<(bb[1]+bb[0])/2:
                        #                 self.lst[-1][0] = "bar"
                        #             else:
                        #                 self.lst[-1][0] = "frac"
                    elif p=="dot":
                        print "dot case"
                        self.lst.pop()
                        dots.append(v)
                        if len(dots)>1 and len(self.lst)>0 and self.lst[-1][0]=="-":
                            l = [dots[-2],self.lst[-1][-1][0],v]
                            image = self.seg.get_combined_strokes(l)
                            bb = self.seg.get_combined_bounding(l)
                            image = self.input_wrapper_arr(image)
                            test = sr.pr(image)
                            p = sr.p(image)
                            probability = sess.run(tf.nn.softmax(test)[0][0][0][p[0]])
                            p = symMap[str(p[0])]
                            print probability,l,p
                            if probability>0.5:
                                self.lst.pop()
                                self.lst.append(["div",bb[0],bb[1],bb[2],bb[3],l])
                                dots.pop()
                                dots.pop()
                        elif len(dots) == 3:
                            image = self.seg.get_combined_strokes(dots)
                            bb = self.seg.get_combined_bounding(dots)
                            image = self.input_wrapper_arr(image)
                            test = sr.pr(image)
                            p = sr.p(image)
                            probability = sess.run(tf.nn.softmax(test)[0][0][0][p[0]])
                            p = symMap[str(p[0])]
                            print probability,dots,p
                            if probability>0.5:
                                self.lst.append(["dots",bb[0],bb[1],bb[2],bb[3],dots])
                                dots = []
                    # elif len(self.lst)>1 and self.lst[-2][0]=="-":
                    #     x = (bb[2]+bb[3])/2
                    #     y = (bb[0]+bb[1])/2
                    #     if x>self.lst[-2][3] and x<self.lst[-2][4]:
                    #         if y>(self.lst[-2][1]+self.lst[-2][2])/2:
                    #             self.lst[-2][0] = "frac"
                    elif p=="x" and len(self.lst)>1 and self.lst[-2][0] in ["a","b","c","d","frac"]:
                        self.lst[-1][0]="mul"

                for w in self.mst[v]:
                    if w[0] in visited:
                        continue
                    queue.append(w[0])

            for e in dots:
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
                image = self.seg.get_combined_strokes(conn)
                bb = self.seg.get_combined_bounding(conn)
                image = self.input_wrapper_arr(image)
                test = sr.pr(image)
                p = sr.p(image)
                probability = sess.run(tf.nn.softmax(test)[0][0][0][p[0]])
                p = symMap[str(p[0])]
                print probability,conn,p
                if probability>0.5 :
                    self.lst.append([p,bb[0],bb[1],bb[2],bb[3],conn])
        self.lst.sort(key = lambda x : x[3])
        print self.lst
        centerList = []
        minusList = []
        deleteList = []
        idx = 0
        for e in self.lst:
            centerList.append([(e[1]+e[2])/2,(e[3]+e[4])/2])
            if e[0]=="-":
                minusList.append(idx)
            idx+=1
        for i in minusList:
            if i in deleteList:
                continue
            up = 0
            down = 0
            k=i-1
            flag = ""
            while centerList[k][1]<self.lst[i][4] and centerList[k][1]>self.lst[i][3]:
                if k in minusList:
                    l = self.lst[i][-1]+self.lst[k][-1]
                    bb = self.seg.get_combined_bounding(l)
                    deleteList.append(i)
                    deleteList.append(k)
                    self.lst.append(["=",bb[0],bb[1],bb[2],bb[3],l])
                    flag = "="
                if centerList[k][0]<centerList[i][0]:
                    up+=1
                else:
                    down+=1
                k-=1
            k=i+1
            while centerList[k][1]<self.lst[i][4] and centerList[k][1]>self.lst[i][3]:
                if k in minusList:
                    l = self.lst[i][-1]+self.lst[k][-1]
                    bb = self.seg.get_combined_bounding(l)
                    deleteList.append(i)
                    deleteList.append(k)
                    self.lst.append(["=",bb[0],bb[1],bb[2],bb[3],l])
                    flag = "="
                if centerList[k][0]<centerList[i][0]:
                    up+=1
                else:
                    down+=1
                k+=1
            if flag == "=":
                continue
            else:
                if up>0:
                    self.lst[i][0] = "frac"
                elif down>0:
                    self.lst[i][0] = "bar"
        for e in deleteList:
            del self.lst[e]




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


if __name__ == '__main__':
    fname='./equations/SKMBT_36317040717260_eq16.png'
    seg = Segmentation(fname)
    d = seg.get_labels()
    mst = MinimumSpanningTree(d).get_mst()
    print mst
    pa = Partition(mst,seg)
    print pa.getList()
    pa.calculateCount()
    print pa.getCount()
    for label in seg.labels.keys():
        print label
        stroke = seg.get_stroke(label)
        scipy.misc.imsave('./tmp/'+ str(label)+'.png', stroke)
