import shelve
import numpy as np
import json
from subprocess import call
import re
import tensorflow as tf
from scipy import misc
from MER_NN import SymbolRecognition, input_wrapper
from os.path import isfile, join
from os import listdir, getcwd, sep

symMap = {}
with open('symbol_mapping.json', 'r') as opened:
    symMap = json.loads(opened.read())
print symMap

model_path = join(getcwd(), "model", "model.ckpt")
sh = shelve.open("training",writeback=False)

with tf.Session() as sess:
    sr = SymbolRecognition(sess, model_path, trainflag=False)
    call(['rm', 'trans/*'])
    call(['rm', 'err/*'])
    count = 0
    acc = 0
    fid = 0
    for i in xrange(1,1000):
        data = sh[str(i)]
        image = data[0]
        res = data[1]
        p = sr.p(image)
        print "predict: ",p
        p = symMap[str(p[0])]
        print "res: ",res,type(res)
        print "p: ", p,type(p)
        misc.imsave('trans/' + str(p) + '_' + res + ".png", np.reshape(image, (32, 32)))
        if p == res:
            acc += 1
        else:
            misc.imsave('err/' + str(p) + '_' + res + ".png",
                        np.reshape(image, (32, 32)))
        fid += 1.
    print(acc, fid, acc / fid)
