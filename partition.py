# TODO
# input mst
# ----------
# dictionary :  key = vertex, value = a list of tuples (tuple: [connected vertex, weight])
# output lst
# ----------
# list :  tuple of format (symbol, x1, y1, x2, y2)
root = -1
value = []
for e in mst:
    root = e
    value = mst[e]
    break
# dp func
# f(v) =
