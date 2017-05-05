import shelve
import json
sh = shelve.open("training",writeback=False)
symDict = {}
count = 0
for i in range(1,3492):
    #and (self.isTriAlphabet(self.sh[str(i)][1])==1
        if (sh[str(i)][1] not in symDict.keys()):
            symDict[sh[str(i)][1]] = count
            count = count + 1




# 38 Symbols, not including s,i,n,c,o,t
print len(symDict.keys())
print symDict


inv_map = {v: k for k, v in symDict.iteritems()}
with open('symbol_mapping.json', 'w') as outfile:
    json.dump(inv_map, outfile)
