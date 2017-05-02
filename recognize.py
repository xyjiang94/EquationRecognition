from MER_NN import SymbolRecognition
import imghdr  # recognize img type
from os import listdir, getcwd, sep
from os.path import isfile, join
from subprocess import call
import re
import json
import tensorflow as tf

symMap = {}
with open('symbol_mapping.json', 'r') as opened:
    symMap = json.loads(opened.read())

model_path = getcwd() + sep + "model"
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
        # do something
        f = join(imgFolderPath, f)
        count += 1
        image = input_wrapper(f)
        p = sr.p(image)
        p = symMap[p[0]]
        g = re.match(".*eq\d*_(.*)_\d*_\d*_\d*_\d*.png", f)
        res = g.group(1)
        misc.imsave('trans/' + str(p) + '_' + res, np.reshape(image, (32, 32)))
        if p == res:
            acc += 1
        else:
            misc.imsave('err/' + str(p) + '_' + res,
                        np.reshape(image, (32, 32)))
        fid += 1.
        if count == 10:
            break
    print(acc, fid, acc / fid)
