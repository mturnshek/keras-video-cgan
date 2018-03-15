import os
import numpy as np
from skimage import color
from PIL import Image

"""
Assumes the model receives the 3 prior frames
to make its prediction for the following frame
"""

DATA_PATH = 'D:/datasets/medium_mario/' # give absolute path


def flatten(list_of_lists):
	return [val for sublist in list_of_lists for val in sublist]


def capture_segment(frames, i):
	x = np.array([frames[i+0], frames[i+1], frames[i+2]])
	y = np.array(frames[i+3])
	return x, y


def get_usable_dataset_from_file(filename):
	path = os.path.abspath(DATA_PATH + filename)
	frames = np.load(path)

	# create the dataset from this file
	# x is array of frames with length 3
	# y is a single frame
	X, Y = [], []
	for i in range(len(frames) - 4):
		x, y = capture_segment(frames, i)
		X.append(x)
		Y.append(y)

	return X, Y

def convert_rgb_channels_to_temporal_grayscale(X):
	"""
	Takes an images array with shape (n, 3, rows, cols, 3),
	and convert it to an array of shape (n, rows, cols, 3),
	where each image is grayscaled and put in one channel
	in the modified images.
	In other words, removes color data and uses those channels
	to stack newly grayscaled temporal images.
	"""
	modified = np.zeros((len(X), len(X[0][0]), len(X[0][0][0]), 3), dtype='uint8')

	for i in range(len(X)):
		images = X[i]
		for t in range(len(images)):
			gray_img = color.rgb2gray(images[t]) * 255
			modified[i, :, :, t] = gray_img.astype('uint8')

	return modified


def load_dataset(grayscale_temporal=False):
	"""
	Load all files in the 'data' directory into a single array.
	Assumes:
		they're all numpy array files with color frame data
		e.g. an array of shape (1233, 110, 127, 3)
		the shapes are the same past the first dimension
	"""
	files = os.listdir(os.path.abspath(DATA_PATH))

	X, Y = [], []
	for file in files:
		X_new, Y_new = get_usable_dataset_from_file(file)
		X.append(X_new)
		Y.append(Y_new)
		print('Loaded file ' + file)

	X, Y = flatten(X), flatten(Y)

	# grayscale the temporal data and put into channels for timesteps
	# X = convert_rgb_channels_to_temporal_grayscale(X)
	# Y = np.array(Y)
	#
	# X = np.pad(X, [0, 18, 1, 0], 'constant')
	# Y = np.pad(Y, [0, 18, 1, 0], constant)

	return np.array(X), np.array(Y)

X, Y = load_dataset(grayscale_temporal=True)
np.save('X.npy', X)
np.save('Y.npy', Y)
