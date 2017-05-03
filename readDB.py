import shelve
import numpy as np
class ReadDB:
    def __init__(self):
        self.sh = shelve.open("training",writeback=False)

    def isTriAlphabet(self,w):
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

    def generateLists(self):
        symDict = {}
        count = 0
        for i in range(1,3483):
            if (self.sh[str(i)][1] not in symDict.keys()) and (self.isTriAlphabet(i)==1):
                symDict[self.sh[str(i)][1]] = count
                count = count + 1




        # 31 Symbols, not including s,i,n,c,o,t
        print len(symDict.keys())

        imageList =[]
        labelsList = []
        for i in range(1,3483):
            if (self.sh[str(i)][1] in symDict.keys()):
                imageList.append(self.sh[str(i)][0])
                #print sh[str(i)][0]
                label = np.zeros(len(symDict.keys()),)
                label[symDict[self.sh[str(i)][1]]] = 1
                #print label
                labelsList.append(label)
        return imageList, labelsList
