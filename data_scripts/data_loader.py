import numpy as np
import os

def load_data():
    sX = np.load(os.path.abspath('D:/datasets/cgan_mario_data/sX.npy'))
    sY = np.load(os.path.abspath('D:/datasets/cgan_mario_data/sY.npy'))
    return sX, sY
