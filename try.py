import numpy as np
x = np.zeros((3,2))
y = np.ones((3,2))
z = x
z = np.concatenate((z,y),axis = 1)
print(z)
list_1 = []
n =10
remain_va = np.arange(0, n, 1) 
list_1 = remain_va.tolist()
list_1.remove(5)
print(list_1)
#for i in range (feature_add):

#print(distance(a,b))
"""
b.remove(feature_add)
for i in range (10):"
	if i not in a:
		print(str(i),"not in a")
"""
