def gen(itr):
    for i in range(0, itr):
        yield i * 2

for x in gen(5):
    print(x)
    break