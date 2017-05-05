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

if __name__ == '__main__':
    generate_eq_latex_mapping()
