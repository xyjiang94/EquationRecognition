import shelve
import imghdr# recognize img type
from os import listdir, getcwd, sep
from os.path import isfile, join
import re
from PIL import Image
import numpy as np

sh = shelve.open("training",writeback=False)
imgFolderPath = getcwd() + sep + "annotated"
files = [f for f in listdir(imgFolderPath) if isfile(join(imgFolderPath, f)) and imghdr.what(join(imgFolderPath, f))=='png']
counter = 0
for e in files:
    h = re.match(".*eq\d*_(.*)_(\d*)_(\d*)_(\d*)_(\d*)\.png",e)
    if h is None:
        print "boss"
    else:
        print "snippet"
        counter+=1
        tmp = []
        symbol = h.group(1)
        y1 = h.group(2)
        y2 = h.group(3)
        x1 = h.group(4)
        x2 = h.group(5)
        with open(join(imgFolderPath, e), 'r+b') as f:
            with Image.open(f) as img_opened:
                img = np.array(img_opened)
                tmp.append(img)
        tmp.append(symbol)
        tmp.append(y1)
        tmp.append(y2)
        tmp.append(x1)
        tmp.append(x2)
        sh[str(counter)] = tmp
        #print tmp
sh.close()
print counter
