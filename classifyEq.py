import json
import math
from partition import *
from segmentation import *
from MinimumSpanningTree import *
from os import listdir, getcwd, sep
from os.path import isfile, join
import imghdr# recognize img type
import tensorflow as tf
import datetime


class Classify(object):

    def __init__(self):
        with open('./lib/eq_symbols.json', 'r') as f:
            self.eq_symbols = json.loads(f.read())
        with open('./lib/eq_latex.json', 'r') as f:
            self.eq_latex = json.loads(f.read())


    # Parameters
    # ----------
    # symbols : list
    #     [u'd', 22, 84, 1002, 1036, [1]]
    #
    # Returns
    # -------
    # list
    #      [equation_number, latex_presentation]
    def classify(self, symbols):
        symbols = self.reform_input(symbols)
        scores = []
        for i in range(1,36):
            score = self.calculate_score(symbols, i)
            scores.append(score)
        m = min(scores)
        index = scores.index(m)
        # print scores
        return [index + 1, self.eq_latex[str(index + 1)]]


    def reform_input(self, symbols):
        d = {}
        for symbol in symbols:
            if symbol[0] not in d:
                d[symbol[0]] = 1
            else:
                d[symbol[0]] += 1
        return d

    def calculate_score(self, symbols, i):
        score = 0.0
        for symbol in symbols:
            if symbol in self.eq_symbols[str(i)]:
                score += math.pow( ( symbols[symbol] - self.eq_symbols[str(i)][symbol] ) , 2) / 2.0
            else:
                score += math.pow( ( symbols[symbol]  ) , 2)

        for symbol in self.eq_symbols[str(i)]:
            if symbol in symbols:
                score += math.pow( ( symbols[symbol] - self.eq_symbols[str(i)][symbol] ) , 2) / 2.0
            else:
                score += math.pow( ( self.eq_symbols[str(i)][symbol]  ) , 2)

        return score



def test_single(fname):
    with tf.Session() as sess:

        model_path = join(getcwd(), "model", "model.ckpt")

        sr = SymbolRecognition(sess, model_path, trainflag=False)

        seg = Segmentation(fname)
        d = seg.get_labels()
        mst = MinimumSpanningTree(d).get_mst()
        pa = Partition(mst,seg,sess,sr)
        l = pa.getList()
        print l
        c = Classify()
        result = c.classify(l)
        print '\n\n\n',result

def test_whole():
    imgFolderPath = getcwd() + sep + "annotated"
    files = [f for f in listdir(imgFolderPath) if isfile(join(imgFolderPath, f)) and imghdr.what(join(imgFolderPath, f))=='png']

    d_count = {}

    model_path = join(getcwd(), "model", "model.ckpt")

    with tf.Session() as sess:
        sr = SymbolRecognition(sess, model_path, trainflag=False)

        count = 0
        for e in files:
            # if count > 100:
            #     break;
            h = re.match(".*eq(\d*).png",e)
            if h is not None:
                count += 1
                print e
                number = h.group(1)
                fname = getcwd() + sep + "annotated" + sep + e
                seg = Segmentation(fname)
                d = seg.get_labels()
                mst = MinimumSpanningTree(d).get_mst()
                pa = Partition(mst,seg,sess,sr)
                l = pa.getList()
                c = Classify()
                result = c.classify(l)

                recognized_symbols = c.reform_input(l)

                d_count[e] = {'image' : e, 'recognized_symbols' : recognized_symbols, 'true' : int(number), 'res':result[0]}
        print d_count

        count = 0
        d_err = []
        for key in d_count:
            if int(d_count[key]['true']) == int(d_count[key]['res']):
                count += 1
            else:
                d_err.append(d_count[key])

        with open('./lib/err.json', 'w') as outfile:
            json.dump(d_err, outfile, indent = 4)
        with open('./lib/res.json', 'w') as outfile:
            json.dump(d_count, outfile, indent = 4)
        print 'accuracy: ', count, len(d_count.keys())

if __name__ == '__main__':
    t1 = datetime.datetime.now()
    test_whole()
    t2 = datetime.datetime.now()
    print t2 - t1

    # test_single('./annotated/SKMBT_36317040717360_eq24.png')
    # test_single('./equations/SKMBT_36317040717260_eq16.png')
