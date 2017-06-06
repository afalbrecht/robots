def makelist():
    return ['a','b','c','d','e']

res = [x for x in range(8)]
del res[4:]
print(res)