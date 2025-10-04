# coding=utf-8
"""
@authors Andrew White
@author: John Mark Mayhall
Optimized: 10/02/2025
This file contains the code that sets up the Unet to modularize the original code.
"""
import keras as k
import tensorflow as tf
from CnnSetupConv import Conv
from keras.layers import Activation, Conv2D, MaxPooling2D, Conv2DTranspose
from keras.layers import BatchNormalization, Dropout, Input, concatenate, Concatenate


class Unet:
    """
    Defines the U-Net architecture.
    """

    def __init__(self, ny: int, nx: int, nvar: int, n_filters: int) -> None:
        """
        Defines needed variables.
        :param ny: Size in y-direction.
        :param nx: Suze in x-direction.
        :param nvar: Number of Variables.
        :param n_filters: Number of filters.
        """
        self.ny = ny
        self.nx = nx
        self.nvar = nvar
        self.n_filters = n_filters
        self.dropout = 0.1
        self.batchnorm = True  # Always true.

    def create_unet(self):
        """
        Creates the U-Net based on the given parameters.
        :return: The created U-Net model.
        """
        concat_axis = 3

        input1 = Input((None, None, 1))
        input2 = Input((None, None, 1))
        inputs = Concatenate()([input1, input2])
        conv_actual = Conv(inputs, self.n_filters * 1, kernel_size=3, batchnorm=self.batchnorm)
        print('Input shape: ', inputs.shape)
        # Contracting Path
        c1 = conv_actual.conv2d_block()
        print('c1: ', c1.shape)
        p1 = MaxPooling2D((2, 2))(c1)
        print('p1 pool: ', p1.shape)
        p1 = Dropout(self.dropout)(p1)
        print('p1 drop: ', p1.shape)

        conv_actual = Conv(p1, self.n_filters * 2, kernel_size=3, batchnorm=self.batchnorm)
        c2 = conv_actual.conv2d_block()
        print('c2: ', c2.shape)
        p2 = MaxPooling2D((2, 2))(c2)
        print('p2 pool: ', p2.shape)
        p2 = Dropout(self.dropout)(p2)
        print('p2 drop: ', p2.shape)

        conv_actual = Conv(p2, self.n_filters * 4, kernel_size=3, batchnorm=self.batchnorm)
        c3 = conv_actual.conv2d_block()
        print('c3: ', c3.shape)
        p3 = MaxPooling2D((2, 2))(c3)
        print('p3 pool: ', p3.shape)
        p3 = Dropout(self.dropout)(p3)
        print('p3 drop: ', p3.shape)

        conv_actual = Conv(p3, self.n_filters * 8, kernel_size=3, batchnorm=self.batchnorm)
        c4 = conv_actual.conv2d_block()
        print('c4: ', c4.shape)
        p4 = MaxPooling2D((2, 2))(c4)
        print('p4 pool: ', p4.shape)
        p4 = Dropout(self.dropout)(p4)
        print('p4 drop: ', p4.shape)

        conv_actual = Conv(p4, self.n_filters * 16, kernel_size=3, batchnorm=self.batchnorm)
        c5 = conv_actual.conv2d_block()
        print('c5: ', c5.shape)
        p5 = MaxPooling2D((2, 2))(c5)
        print('p4 pool: ', p5.shape)
        p5 = Dropout(self.dropout)(p5)
        print('p4 drop: ', p5.shape)

        conv_actual = Conv(p5, self.n_filters * 1, kernel_size=3, batchnorm=self.batchnorm)
        c6 = conv_actual.conv2d_block()
        print('c6: ', c6.shape)
        # The expansive path starts below.

        u7 = Conv2DTranspose(self.n_filters * 16, (3, 3), strides=(2, 2), padding='same')(c6)
        print('u7: ', u7.shape)
        u7 = concatenate([u7, c5], axis=concat_axis)
        print('u7 cat: ', u7.shape)
        u7 = Dropout(self.dropout)(u7)
        conv_actual = Conv(u7, self.n_filters * 16, kernel_size=3, batchnorm=self.batchnorm)
        c7 = conv_actual.conv2d_block()
        print('c7: ', c7.shape)

        u8 = Conv2DTranspose(self.n_filters * 8, (3, 3), strides=(2, 2), padding='same')(c7)
        print('u8: ', u8.shape)
        u8 = concatenate([u8, c4], axis=concat_axis)
        print('u8 cat: ', u8.shape)
        u8 = Dropout(self.dropout)(u8)
        conv_actual = Conv(u8, self.n_filters * 8, kernel_size=3, batchnorm=self.batchnorm)
        c8 = conv_actual.conv2d_block()
        print('c8: ', c8.shape)

        u9 = Conv2DTranspose(self.n_filters * 4, (3, 3), strides=(2, 2), padding='same')(c8)
        print('u9: ', u9.shape)
        u9 = concatenate([u9, c3], axis=concat_axis)
        print('u9 cat: ', u9.shape)
        u9 = Dropout(self.dropout)(u9)
        conv_actual = Conv(u9, self.n_filters * 4, kernel_size=3, batchnorm=self.batchnorm)
        c9 = conv_actual.conv2d_block()
        print('c9: ', c9.shape)

        u10 = Conv2DTranspose(self.n_filters * 2, (3, 3), strides=(2, 2), padding='same')(c9)
        print('u10: ', u10.shape)
        u10 = concatenate([u10, c2], axis=concat_axis)
        print('u10 cat: ', u10.shape)
        u10 = Dropout(self.dropout)(u10)
        conv_actual = Conv(u10, self.n_filters * 2, kernel_size=3, batchnorm=self.batchnorm)
        c10 = conv_actual.conv2d_block()
        print('c10: ', c10.shape)

        u11 = Conv2DTranspose(self.n_filters * 1, (3, 3), strides=(2, 2), padding='same')(c10)
        print('u11: ', u11.shape)
        u11 = concatenate([u11, c1], axis=concat_axis)
        print('u11 cat: ', u11.shape)
        u11 = Dropout(self.dropout)(u11)
        conv_actual = Conv(u11, self.n_filters * 1, kernel_size=3, batchnorm=self.batchnorm)
        c11 = conv_actual.conv2d_block()
        print('c11: ', c11.shape)

        outputs = Conv2D(1, (1, 1), activation='sigmoid')(c11)
        model = k.Model(inputs=(input1, input2), outputs=outputs)
        return model
