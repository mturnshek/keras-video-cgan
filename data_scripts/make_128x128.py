import os
import numpy as np
from PIL import Image

pX = np.load('pX.npy')
pY = np.load('pY.npy')

sX = pX[:, 96:224, 64:192, :]
sY = pY[:, 96:224, 64:192, :]

img = Image.fromarray(sX[1000])
img.show()

np.save('sX.npy', sX)
np.save('sY.npy', sY)
