import shelve
import numpy as np
sh = shelve.open("training",writeback=False)

def isTriAlphabet(w):
    if w == 's':
        return 0
    elif w == 'i':
        return 0
    elif w == 'n':
        return 0
    elif w == 'o':
        return 0
    elif w == 't':
        return 0
    else:
        return 1


symDict = {}
count = 0
for i in range(1,3483):
    if (sh[str(i)][1] not in symDict.keys()) and (isTriAlphabet(i)==1):
        symDict[sh[str(i)][1]] = count
        count = count + 1




# 31 Symbols, not including s,i,n,c,o,t
print len(symDict.keys())

imageList =[]
labelsList = []
for i in range(1,3483):
    if (sh[str(i)][1] in symDict.keys()):
        imageList.append(sh[str(i)][0])
        #print sh[str(i)][0]
        label = np.zeros(len(symDict.keys()),)
        label[symDict[sh[str(i)][1]]] = 1
        #print label
        labelsList.append(label)







