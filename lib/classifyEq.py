import json
import math

class Classify(object):

    def __init__(self):
        with open('eq_symbols.json', 'r') as f:
            self.eq_symbols = json.loads(f.read())


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
        print scores
        return [index + 1, self.eq_symbols[str(index + 1)]]


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

if __name__ == '__main__':
    symbols = [[u'd', 22, 84, 1002, 1036, [1]], [u'a', 48, 78, 959, 1005, [5]], ['frac', 90, 101, 942, 1065, [7]], [u'b', 118, 163, 964, 995, [20]], [u'c', 129, 165, 1013, 1059, [23]], ['=', 98, 119, 855, 894, [[13], 17]], ['frac', 99, 111, 739, 798, [14]], [u'c', 126, 164, 750, 793, [22]], [u'd', 35, 93, 745, 780, [2]], [u'2', 93, 125, 657, 690, [9]], [u'b', 116, 159, 584, 612, [19]], ['frac', 91, 99, 560, 625, [8]], [u'a', 48, 81, 568, 613, [4]], ['=', 95, 114, 489, 523, [[12], 15]], ['frac', 95, 108, 379, 437, [11]], [u'c', 47, 84, 385, 427, [3]], [u'd', 119, 175, 384, 420, [21]], [u'dot', 131, 142, 306, 315, [24]], ['frac', 113, 122, 289, 326, [18]], [u'dot', 94, 103, 303, 313, [10]], ['frac', 108, 117, 176, 245, [16]], [u'a', 62, 98, 186, 235, [6]], [u'b', 136, 175, 192, 229, [25]]]
    c = Classify()
    result = c.classify(symbols)
    print '\n\n\n',result
