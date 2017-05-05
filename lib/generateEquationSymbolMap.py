import json
import re

def add_symbol(d,symbol):
    if symbol in d:
        d[symbol] += 1
    else:
        d[symbol] = 1

with open('symbol_mapping_all.json', 'r') as opened:
    symMap = json.loads(opened.read())

with open('eq.txt','r') as eq:
    lines = eq.readlines()

d_eqSym = {}

i = 0
for line in lines:
    i += 1
    d_eqSym[i] = {}
    eq_symbols = []
    if re.match(r'.*?\\frac{.*?}{.*?}.*?',line):

        fracs = re.findall(r'\\frac{.*?}{.*?}',line)
        for exp in fracs:
            exp = re.sub(r'\\frac{','',exp)
            exp = re.sub(r'}','',exp)
            symbols = re.split('{',exp)
            eq_symbols.extend(symbols)
            eq_symbols.append('frac')
        line = re.sub(r'\\frac{.*?}{.*?}','',line)

    if re.match(r'\\sqrt',line):
        sqrts = re.findall(r'\\sqrt',line)
        for exp in sqrts:
            eq_symbols.append('sqrt')
        line = re.sub(r'\\sqrt','',line)

    for c in line:
        if c != '\n':
            eq_symbols.append(c)




    for symbol in eq_symbols:
        add_symbol(d_eqSym[i], symbol)

print d_eqSym[12]

with open('eq_symbols1.json', 'w') as outfile:
    json.dump(d_eqSym, outfile, indent = 4)

# # sh = shelve.open("eq",writeback=False)
# #
# # for i in l_Latex:
# #     print i
# # print len(l_Latex)
