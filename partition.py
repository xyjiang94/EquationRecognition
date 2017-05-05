# TODO
# input mst
# ----------
# dictionary :  key = vertex, value = a list of tuples (tuple: [connected vertex, weight])
# output lst
# ----------
# list :  tuple of format (symbol, x1, y1, x2, y2)
from segmentation import Segmentation
from MinimumSpanningTree import MinimumSpanningTree

class Partition(object):
    def __init__(self,mst):
        self.mst = mst
        self.lst = []

    def getList():
        return self.lst

    def generateList():
        pass


fname='./equations/SKMBT_36317040717260_eq6.png'
seg = Segmentation(fname)
d = seg.get_labels()
print d
mst = MinimumSpanningTree(d)
pa = Partition(mst)
