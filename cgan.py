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
import numpy as np
import os

from data_scripts.data_loader import load_data

class Pix2Pix():
    def __init__(self, load_weights=True):
        # Input shape
        self.img_rows = 128
        self.img_cols = 128
        self.channels = 3
        self.img_shape = (self.img_rows, self.img_cols, self.channels)

        self.data_B, self.data_A = load_data()
        # Normalize pixels to be values between -1 and 1
        self.data_A = self.data_A/127.5 - 1
        self.data_B = self.data_B/127.5 - 1

        # Calculate output shape of D (PatchGAN)
        patch = int(self.img_rows / 2**4)
        self.disc_patch = (patch, patch, 1)
        # this is basically the discriminator taking each 16x16 part
        # of the image and condensing it into a single 'real' or 'fake'.

        # Number of filters in the first layer of G and D
        self.gf = 64
        self.df = 64

        self.g_trains_per_d = 3
        optimizer = Adam(0.0002, 0.5)

        # Build and compile the discriminators
        self.discriminator1 = self.build_discriminator()
        self.discriminator1.compile(loss='mse',
            optimizer=optimizer,
            metrics=['accuracy'])

        # Two discriminators make the generations closer to the actual data
        # instead of them preying on the blind spot of one disciminator
        self.discriminator2 = self.build_discriminator()
        self.discriminator2.compile(loss='mse',
            optimizer=optimizer,
            metrics=['accuracy'])

        # Build and compile the generator
        self.generator = self.build_generator()
        self.generator.compile(loss='binary_crossentropy', optimizer=optimizer)

        if load_weights:
            self.generator.load_weights("saved_model/generator.hdf5")
            self.discriminator1.load_weights("saved_model/discriminator1.hdf5")
            self.discriminator2.load_weights("saved_model/discriminator2.hdf5")

        # Input images and their conditioning images
        img_A = Input(shape=self.img_shape)
        img_B = Input(shape=self.img_shape)

        # By conditioning on B generate a fake version of A
        fake_A = self.generator(img_B)

        # For the combined model we will only train the generator
        self.discriminator1.trainable = False
        self.discriminator2.trainable = False

        # Discriminators determines validity of translated images / condition pairs
        valid1 = self.discriminator1([fake_A, img_B])
        valid2 = self.discriminator2([fake_A, img_B])

        self.combined1 = Model([img_A, img_B], [valid1, fake_A])
        self.combined1.compile(loss=['mse', 'mae'],
                              loss_weights=[1, 100],
                              optimizer=optimizer)

        self.combined2 = Model([img_A, img_B], [valid2, fake_A])
        self.combined2.compile(loss=['mse', 'mae'],
                              loss_weights=[1, 100],
                              optimizer=optimizer)

    def save_models(self):
        self.generator.save_weights('saved_model/generator.hdf5')
        self.discriminator1.save_weights('saved_model/discriminator1.hdf5')
        self.discriminator2.save_weights('saved_model/discriminator2.hdf5')

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

    def sample(self, batch_size):
        # Sample images and their conditioning counterparts
        # TODO: sample without replacement
        indices = np.random.randint(0, self.data_A.shape[0], batch_size)
        imgs_A, imgs_B = self.data_A[indices], self.data_B[indices]
        fake_A = self.generator.predict(imgs_B) # gen fakes from imgs_B

        return imgs_A, imgs_B, fake_A

    def train(self, epochs, batch_size=5, save_interval=50):

        start_time = datetime.datetime.now()

        for epoch in range(epochs):

            real = np.ones((batch_size,) + self.disc_patch)
            fake = np.zeros((batch_size,) + self.disc_patch)

            # ----------------------
            #  Train Discriminators
            # ----------------------

            imgs_A, imgs_B, fake_A = self.sample(batch_size)
            # Train the first discriminator
            d1_loss_real = self.discriminator1.train_on_batch([imgs_A, imgs_B], real)
            d1_loss_fake = self.discriminator1.train_on_batch([fake_A, imgs_B], fake)
            d1_loss = 0.5 * np.add(d1_loss_real, d1_loss_fake)

            imgs_A, imgs_B, fake_A = self.sample(batch_size)
            # Train the second discriminator
            d2_loss_real = self.discriminator2.train_on_batch([imgs_A, imgs_B], real)
            d2_loss_fake = self.discriminator2.train_on_batch([fake_A, imgs_B], fake)
            d2_loss = 0.5 * np.add(d2_loss_real, d2_loss_fake)

            # ------------------
            #  Train Generator
            # ------------------

            imgs_A, imgs_B, _ = self.sample(batch_size)
            # Use a given discriminator with 50% chance each time.
            # Sort of like how SGD tends to stabilize, I assume this will as well.
            if (np.random.random() > 0.5):
                g_loss = self.combined1.train_on_batch([imgs_A, imgs_B], [real, imgs_A])
            else:
                g_loss = self.combined2.train_on_batch([imgs_A, imgs_B], [real, imgs_A])

            # ----------------
            #   Log progress
            # ----------------

            elapsed_time = datetime.datetime.now() - start_time
            print ("%d time: %s" % (epoch, elapsed_time))
            print ("g loss", g_loss, "\n", "d1 loss:", d1_loss, "d2_loss:", d2_loss)

            # If at save interval => save generated image samples and models
            if epoch % save_interval == 0:
                self.save_imgs(epoch)
                self.save_models()

    def save_imgs(self, epoch):
        os.makedirs('images/%s' % 'mario', exist_ok=True)
        r, c = 3, 3

        batch_size = 3
        indices = np.random.randint(0, self.data_A.shape[0], batch_size)
        imgs_A, imgs_B = self.data_A[indices], self.data_B[indices]

        fake_A = self.generator.predict(imgs_B)

        gen_imgs = np.concatenate([imgs_B, fake_A, imgs_A])

        # Rescale images 0 - 255
        gen_imgs = (gen_imgs + 1) * 127.5

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
    gan = Pix2Pix(load_weights=False)
    gan.train(epochs=999999, batch_size=8, save_interval=200)
