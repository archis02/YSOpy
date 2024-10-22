import numpy as np
import corner
import matplotlib.pyplot as plt

arr = np.load("params_1.npy")
x = np.zeros(shape=(1800,9))
for i in range(100):
    for j in range(18):
        x[i*18+j] = arr[i,j,:]

# print(x.shape)
# print(x)
# print("shape",arr.shape)

figure = corner.corner(
    x,
    labels=[
        r'$M$',
        r'log $\dot M$',
        r'$B$', r'$\theta$',
        r'log $n_e$',
        r'$R_{\ast}$',
        r'$T_o$',
        r'$T_{slab}$',
        r'$\tau$'
    ],
    show_titles=True
)
plt.show()
#print(arr)