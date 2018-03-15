import numpy as np

Y = np.load('Y.npy')
pY = np.pad(Y, ((0, 0), (18, 18), (1, 1), (0, 0)), 'constant')

np.save('pY.npy', pY)
