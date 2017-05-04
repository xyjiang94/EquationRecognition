import shelve
sh = shelve.open("training",writeback=False)
for i in range(1,10):
    print sh[str(i)]
