from readDB import *
import json
import numpy as np
import re

class myarray(np.ndarray):
    def __new__(cls, *args, **kwargs):
        return np.array(*args, **kwargs).view(myarray)
    def index(self, value):
        return np.where(self==value)

sh = shelve.open("training",writeback=False)
db = ReadDB()
x,y = db.generateLists()


symMap = {}
with open('symbol_mapping.json', 'r') as opened:
    symMap = json.loads(opened.read())
print symMap


list=[]
for i in range(len(x)):
    for j in range(1, 3483):
        #(x[i]==sh[str(j)][0]).all()
        if (np.array_equal(x[i], sh[str(j)][0])):
            print x[i].shape
            print sh[str(j)][0].shape
            print i
            print j
            a = myarray(y[i])
            k=a.index(1)
            print k[0][0]
            print type(k[0][0])
            a = symMap[str(k[0][0])]
            b = sh[str(j)][1]
            if(a==b):
                list.append(1)
            else:
                list.append(0)


thefile = open('test.txt', 'w')
for item in list:
  thefile.write("%s\n" % item)
thefile.close()