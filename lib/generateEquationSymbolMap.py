import json
import re
from os import listdir, getcwd, sep
from os.path import isfile, join
import imghdr# recognize img type


def add_symbol(d,number,symbol):
    if number not in d:
        d[number] = {}
    d_eq = d[number]
    if symbol in d_eq:
        d_eq[symbol] += 1
    else:
        d_eq[symbol] = 1

def generate_eq_symbol_mapping():
    imgFolderPath = getcwd() + sep + "annotated"
    files = [f for f in listdir(imgFolderPath) if isfile(join(imgFolderPath, f)) and imghdr.what(join(imgFolderPath, f))=='png']

    d_eqSym = {}

    for e in files:
        h = re.match(".*eq(\d*)_(.*)_(\d*)_(\d*)_(\d*)_(\d*)\.png",e)
        if h is None:
            print "equation"
        else:
            print "snippet"
            number = h.group(1)
            symbol = h.group(2)
            add_symbol(d_eqSym,number,symbol)

    with open('eq_symbols.json', 'w') as outfile:
        json.dump(d_eqSym, outfile, indent = 4)

def generate_eq_latex_mapping():
    with open('eq.txt','r') as eq:
        lines = eq.readlines()
        d_eqLatex = {}
        i = 0
        for line in lines:
            line = re.sub(r'\n','',line)
            i += 1
            d_eqLatex[i] = line

    with open('eq_latex.json', 'w') as outfile:
        json.dump(d_eqLatex, outfile, indent = 4)

def modify_err():
    with open('./lib/err.json', 'r') as f:
        err = json.loads(f.read())
    with open('./lib/eq_symbols.json', 'r') as f:
        eq_symbols = json.loads(f.read())

    for d in err:
        print d
        d['true_symbols'] = eq_symbols[str(d['true'])]

    with open('./lib/err.json', 'w') as outfile:
        json.dump(err, outfile, indent = 4)

def err_analyze():
    with open('./lib/err.json', 'r') as f:
        err = json.loads(f.read())
        d_anly = {}
        for d in err:
            if d['true'] not in d_anly:
                d_anly[d['true']] = {'count': 0, 'images' : []}
            d_anly[d['true']]['images'].append(d['image'])
            d_anly[d['true']]['count'] += 1
        with open('./lib/err_analyze.json', 'w') as outfile:
            json.dump(d_anly, outfile, indent = 4)

        l = []
        for key in d_anly:
            l.append([key, d_anly[key]['count']])

        l = sorted(l, key = lambda x: x[1])
        print l


if __name__ == '__main__':
    # generate_eq_latex_mapping()
    # modify_err()
    err_analyze()
