# coding=utf-8
"""
@authors Andrew White
@author: John Mark Mayhall
Optimized: 10/02/2025
"""
from keras.layers import Activation, Conv2D, MaxPooling2D, Conv2DTranspose
from keras.layers import BatchNormalization, Dropout, Input, concatenate


class Conv:
    """
    Class for setting up the convolutional layers.
    """

    def __init__(self, input_tensor, n_filters: int, kernel_size: int,
                 batchnorm: bool) -> None:
        """
        Defines needed variables.
        :param input_tensor: The inputted tensor to be used for the convolutional layers.
        :param n_filters: Number of filters.
        :param kernel_size: Size of the kernel.
        :param batchnorm: Determines if batch normalization will be used, always True.
        """
        self.input_tensor = input_tensor
        self.n_filters = n_filters
        self.kernel_size = kernel_size
        self.batchnorm = batchnorm

    def conv2d_block(self):
        """
        Function to add two convolutional layers with the parameters passed to it.
        :return: The two convolutional layers.
        """
        # first layer
        x = Conv2D(filters=self.n_filters, kernel_size=(self.kernel_size, self.kernel_size),
                   kernel_initializer='he_normal', padding='same')(self.input_tensor)
        if self.batchnorm:
            x = BatchNormalization()(x)
        x = Activation('relu')(x)
        print('2 Layer conv 1: ', x.shape)

        # second layer
        x = Conv2D(filters=self.n_filters, kernel_size=(self.kernel_size, self.kernel_size),
                   kernel_initializer='he_normal', padding='same')(self.input_tensor)
        if self.batchnorm:
            x = BatchNormalization()(x)
        x = Activation('relu')(x)
        print('2 Layer conv 2: ', x.shape)

        return x
