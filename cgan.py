from __future__ import print_function, division
import scipy

from keras.datasets import mnist
from keras_contrib.layers.normalization import InstanceNormalization
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, Concatenate
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam
import datetime
import matplotlib.pyplot as plt
import sys
from data_loader import DataLoader
import numpy as np
import os

from manage_data import load_data

class Pix2Pix():
    def __init__(self, load_weights=True):
        # Input shape
        self.img_rows = 128
        self.img_cols = 128
        self.channels = 3
        self.img_shape = (self.img_rows, self.img_cols, self.channels)

        # Configure data loader
        # self.dataset_name = 'facades'
        # self.data_loader = DataLoader(dataset_name=self.dataset_name,
        #                               img_res=(self.img_rows, self.img_cols))

        self.data_A, self.data_B = load_data()
        self.data_A = self.data_A/127.5 - 1
        self.data_B = self.data_B/127.5 - 1

        # Calculate output shape of D (PatchGAN)
        patch = int(self.img_rows / 2**4)
        self.disc_patch = (patch, patch, 1)

        # Number of filters in the first layer of G and D
        self.gf = 64
        self.df = 64

        optimizer = Adam(0.0002, 0.5)

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='mse',
            optimizer=optimizer,
            metrics=['accuracy'])

        # Build and compile the generator
        self.generator = self.build_generator()
        self.generator.compile(loss='binary_crossentropy', optimizer=optimizer)
        # M: does the loss of this generator and its optimizer actually matter?
        # M: it seems like it will never be trained by itself, no?

        # M :
        if load_weights:
            self.generator.load_weights("saved_model/generator.hdf5")
            self.discriminator.load_weights("saved_model/discriminator.hdf5")

        # Input images and their conditioning images
        img_A = Input(shape=self.img_shape)
        img_B = Input(shape=self.img_shape)

        # By conditioning on B generate a fake version of A
        fake_A = self.generator(img_B)
        # M: i don't quite understand what this line does ...
        # M: self.generator is a model. by passing it img_B, which is an input ...
        # M: i'm going to keep going for now and come back.

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # Discriminators determines validity of translated images / condition pairs
        valid = self.discriminator([fake_A, img_B])

        self.combined = Model([img_A, img_B], [valid, fake_A])
        self.combined.compile(loss=['mse', 'mae'],
                              loss_weights=[1, 100],
                              optimizer=optimizer)

    def save_models(self):
        self.generator.save_weights('saved_model/generator.hdf5')
        self.discriminator.save_weights('saved_model/discriminator.hdf5')

    def build_generator(self):
        """U-Net Generator"""

        def conv2d(layer_input, filters, f_size=4, bn=True):
            """Layers used during downsampling"""
            d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
            d = LeakyReLU(alpha=0.2)(d)
            if bn:
                d = BatchNormalization(momentum=0.8)(d)
            return d

        def deconv2d(layer_input, skip_input, filters, f_size=4, dropout_rate=0):
            """Layers used during upsampling"""
            u = UpSampling2D(size=2)(layer_input)
            u = Conv2D(filters, kernel_size=f_size, strides=1, padding='same', activation='relu')(u)
            if dropout_rate:
                u = Dropout(dropout_rate)(u)
            u = BatchNormalization(momentum=0.8)(u)
            u = Concatenate()([u, skip_input])
            return u

        # Image input
        d0 = Input(shape=self.img_shape)

        # Downsampling
        d1 = conv2d(d0, self.gf, bn=False)
        d2 = conv2d(d1, self.gf*2)
        d3 = conv2d(d2, self.gf*4)
        d4 = conv2d(d3, self.gf*8)
        d5 = conv2d(d4, self.gf*8)
        d6 = conv2d(d5, self.gf*8)
        # d7 = conv2d(d6, self.gf*8)

        # Upsampling
        # u1 = deconv2d(d7, d6, self.gf*8)
        u2 = deconv2d(d6, d5, self.gf*8)
        u3 = deconv2d(u2, d4, self.gf*8)
        u4 = deconv2d(u3, d3, self.gf*4)
        u5 = deconv2d(u4, d2, self.gf*2)
        u6 = deconv2d(u5, d1, self.gf)

        u7 = UpSampling2D(size=2)(u6)
        output_img = Conv2D(self.channels, kernel_size=4, strides=1, padding='same', activation='tanh')(u7)

        return Model(d0, output_img)

    # M: the generator seems to be fully deterministic - no noise input.
    # M: why? what would the effects of adding noise be?
    # M: would it be optimal for A) mario B) MCTSGAN
    # M: it appears that in both cases, noise would actually be better.
    # M: the way that this is being used now seems more like
    # M: deterministic translation, rather than data point generation.

    def build_discriminator(self):

        def d_layer(layer_input, filters, f_size=4, bn=True):
            """Discriminator layer"""
            d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
            d = LeakyReLU(alpha=0.2)(d)
            if bn:
                d = BatchNormalization(momentum=0.8)(d)
            return d

        img_A = Input(shape=self.img_shape)
        img_B = Input(shape=self.img_shape)

        # Concatenate image and conditioning image by channels to produce input
        combined_imgs = Concatenate(axis=-1)([img_A, img_B])

        d1 = d_layer(combined_imgs, self.df, bn=False)
        d2 = d_layer(d1, self.df*2)
        d3 = d_layer(d2, self.df*4)
        d4 = d_layer(d3, self.df*8)

        validity = Conv2D(1, kernel_size=4, strides=1, padding='same')(d4)

        return Model([img_A, img_B], validity)

    # batch size was 1
    def train(self, epochs, batch_size=5, save_interval=50):

        start_time = datetime.datetime.now()

        for epoch in range(epochs):

            # ----------------------
            #  Train Discriminator
            # ----------------------

            # Sample images and their conditioning counterparts
            indices = np.random.randint(0, self.data_A.shape[0], batch_size)
            imgs_A, imgs_B = self.data_A[indices], self.data_B[indices]
            # imgs_A, imgs_B = self.data_loader.load_data(batch_size)

            # Condition on B and generate a translated version
            fake_A = self.generator.predict(imgs_B)

            valid = np.ones((batch_size,) + self.disc_patch)
            fake = np.zeros((batch_size,) + self.disc_patch)

            # Train the discriminators (original images = real / generated = Fake)
            d_loss_real = self.discriminator.train_on_batch([imgs_A, imgs_B], valid)
            d_loss_fake = self.discriminator.train_on_batch([fake_A, imgs_B], fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # ------------------
            #  Train Generator
            # ------------------

            # Sample images and their conditioning counterparts
            indices = np.random.randint(0, self.data_A.shape[0], batch_size)
            imgs_A, imgs_B = self.data_A[indices], self.data_B[indices]
            # imgs_A, imgs_B = self.data_loader.load_data(batch_size)

            # The generators want the discriminators to label the generated images as real
            valid = np.ones((batch_size,) + self.disc_patch)

            # Train the generators
            g_loss = self.combined.train_on_batch([imgs_A, imgs_B], [valid, imgs_A])

            elapsed_time = datetime.datetime.now() - start_time
            # Plot the progress
            print ("%d time: %s" % (epoch, elapsed_time))
            print ("g loss:", g_loss, "d loss:", d_loss)

            # If at save interval => save generated image samples
            if epoch % save_interval == 0:
                self.save_imgs(epoch)
                self.save_models()

    def save_imgs(self, epoch):
        os.makedirs('images/%s' % 'mario', exist_ok=True)
        r, c = 3, 3

        batch_size = 3
        indices = np.random.randint(0, self.data_A.shape[0], batch_size)
        imgs_A, imgs_B = self.data_A[indices], self.data_B[indices]

        # imgs_A, imgs_B = self.data_loader.load_data(batch_size=3, is_testing=True)
        fake_A = self.generator.predict(imgs_B)

        gen_imgs = np.concatenate([imgs_B, fake_A, imgs_A])

        # Rescale images 0 - 255
        gen_imgs = (gen_imgs + 1) * 127.5
        # gen_imgs = 0.5 * gen_imgs + 0.5

        titles = ['Condition', 'Generated', 'Original']
        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(gen_imgs[cnt].astype('uint8'))
                axs[i, j].set_title(titles[i])
                axs[i,j].axis('off')
                cnt += 1
        fig.savefig("images/%s/%d.png" % ('mario', epoch))
        plt.close()


if __name__ == '__main__':
    gan = Pix2Pix(load_weights=True)
    # batch size was 1
    gan.train(epochs=999999, batch_size=8, save_interval=200)
