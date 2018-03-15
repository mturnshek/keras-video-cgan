import numpy as np
from skimage import color

X = np.load('X.npy')

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

np.save('gX.npy', convert_rgb_channels_to_temporal_grayscale(X))
