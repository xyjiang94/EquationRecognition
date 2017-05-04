import imghdr  # recognize img type
from os import listdir, getcwd, sep
from os.path import isfile, join
from subprocess import call
import re
import json
import tensorflow as tf
from scipy import misc
import numpy as np
from MER_NN import SymbolRecognition, input_wrapper

symMap = {}
with open('symbol_mapping.json', 'r') as opened:
    symMap = json.loads(opened.read())
print symMap

model_path = join(getcwd(), "model", "model.ckpt")
imgFolderPath = getcwd() + sep + "annotated"
files = [f for f in listdir(imgFolderPath) if isfile(
    join(imgFolderPath, f)) and imghdr.what(join(imgFolderPath, f)) == 'png']

with tf.Session() as sess:
    sr = SymbolRecognition(sess, model_path, trainflag=False)
    call(['rm', 'trans/*'])
    call(['rm', 'err/*'])
    count = 0
    acc = 0
    fid = 0
    for f in files:
        # recognize files
        the_file = f
        g = re.match(".*eq\d*_(.*)_\d*_\d*_\d*_\d*.png", the_file)
        if g:
            res = g.group(1)
        else:
            continue
        f = join(imgFolderPath, f)
        count += 1
        image = input_wrapper(f)
        # test = sr.pr(image)
        # print "test: ",test
        # print sess.run(tf.nn.softmax(test),dim=-1)
        p = sr.p(image)
        print "predict: ",p
        p = symMap[str(p[0])]
        print the_file

        print "res: ",res,type(res)
        print "p: ", p,type(p)
        misc.imsave('trans/' + str(p) + '_' + res + ".png", np.reshape(image, (32, 32)))
        if p == res:
            acc += 1
        else:
            misc.imsave('err/' + str(p) + '_' + res + ".png",
                        np.reshape(image, (32, 32)))
        fid += 1.
        if count >= 1000:
            break
    print(acc, fid, acc / fid)
