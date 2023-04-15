import numpy as np

# arr = np.array([10, 15, 20, 25, 30, 35, 40])
# print(arr[1:4])

a = np.arange(20)
# print(a)

d = np.empty(5)
# print(d)

arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
newarr = arr.reshape(4, 3)
# print(newarr)

arr = np.array([1, 2, 3])
for idx, x in np.ndenumerate(arr):
    pass
#   print(idx)
#   print(x)

f = np.ones(5)
# print(f)

g = np.random.rand(5)
# print(g)

g = g * 6 + 1
g = g.astype(np.int8)
# print(g)

a = np.arange(1, 101).reshape(10, 10)
print(a[ 0 , 0 ])             # 0th only 
print(a[ 1 , 1 ])             # 1st only 
print(a[ 1 : 3 , 1 : 3 ])         # 1st to 2nd 
print(a[ 1 : 5 : 3 , 1 : 5 : 3 ])     # 1st to 4th (in increments of 3) 
print(a[: 5 , : 5 ])           # 1st to 4th 
print(a[ 7 :, 7 :])           # from the 7th to the end
print(a[:: 3 , :: 3 ])         # from top to bottom (in increments of 3) 
print(a[:, :])             # all over (top to bottom)"

# Create ufunc

def myadd(x, y):
  return x+y

myadd = np.frompyfunc(myadd, 2, 1)

print(myadd([1, 2, 3, 4], [5, 6, 7, 8]))
