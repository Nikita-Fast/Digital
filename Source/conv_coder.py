import numpy as np


shr_len = 7
p = np.array([[1,1,1,1,0,0,1], [1,0,1,1,0,1,1]])
np.random.seed(666)
bits = np.random.randint(0,2,10)
print(bits)
iters = len(bits) - shr_len + 1
res = -1*np.ones(2*iters)


for i in range(iters):
    out = bits[i: i+shr_len] @ p.T
    print(i, bits[i: i+shr_len], out)
    res[2*i: 2*i+2] = out

print(res)