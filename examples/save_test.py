import numpy as np
import matplotlib.pyplot as plt
a = []

for i in range(100):
    a.append(i)

b = np.array(a)

np.save('r',b)


y = np.load('r.npy')
x = np.arange(0,len(y))

plt.plot(x,y)
plt.show()


