# coding=utf-8
"""
@authors Andrew White
@author: John Mark Mayhall
Optimized: 10/02/2025
Class to help finalize the model by saving needed files and calculating needed variables.
"""
import keras as k
import numpy as np
import tensorflow as tf
from CnnSetupUnet import Unet
from sklearn.utils.class_weight import compute_class_weight


def dice_loss(y_true, y_pred) -> int:
    """
    Custom dice loss calculation function
    :param y_true: Truth array
    :param y_pred: Prediction
    :return: Loss value
    """
    loss = 1 - (2 * tf.reduce_sum(y_true * y_pred)) / (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred))
    return loss


def custom_weighted_binary_crossentropy(zero_weight, one_weight):
    """
    Custom loss function based on a weighted binary crossentropy
    :param zero_weight: Weight of null values
    :param one_weight: Weight of true values
    :return: Weighted loss function results
    """

    def weighted_binary_crossentropy(y_true, y_pred):
        """
        Function to calculate the weighted binary crossentropy
        :param y_true: Truth array
        :param y_pred: Prediction array
        :return: Loss function value
        """
        y_true = k.ops.cast(y_true, dtype=tf.float32)

        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)

        # Compute cross entropy from probabilities.
        bce = y_true * tf.math.log(y_pred + epsilon)
        bce += (1 - y_true) * tf.math.log(1 - y_pred + epsilon)
        bce = -bce

        # Apply the weights to each class individually
        weight_vector = y_true * one_weight + (1. - y_true) * zero_weight
        weighted_bce = weight_vector * bce

        # Return the mean error
        return tf.reduce_mean(weighted_bce)

    return weighted_binary_crossentropy


class Final:
    """
    Finalize U-Net model.
    """

    def __init__(self, xsize: int, ysize: int, nvars: int, batch_size: int, epochs: int, x_train,
                 y_train, x_val, y_val, x_train8, x_val8):
        """
        Defines needed variables.
        :param xsize: Size in x-direction.
        :param ysize: Size in y-direction.
        :param nvars: Number of variables.
        :param batch_size: Number of arrays per batch.
        :param epochs: Number of epochs to be run.
        :param x_train: Channel 13 training arrays.
        :param y_train: Geojson training arrays.
        :param x_val: Channel 13 validation arrays.
        :param y_val: Geojson validation arrays.
        :param x_train8: Channel 8 validation arrays.
        :param x_val8: Channel 8 validation arrays.
        """
        self.xsize, self.ysize = xsize, ysize
        self.nvars, self.batch_size, self.epochs = nvars, batch_size, epochs
        self.x_train, self.y_train = x_train, y_train
        self.x_val, self.y_val = x_val, y_val
        self.x_train8, self.x_val8 = x_train8, x_val8

    def create_all(self):
        """
        Runs the functions that will create the unet and convolutional layers
        :return: The final U-Net model that has been created.
        """

        net = Unet(self.ysize, self.xsize, self.nvars, n_filters=32)  # Defines the variable the class is equal to.
        unet = net.create_unet()  # Creates the unet from the CnnSetupUnet file.

        y_integers = self.y_train.flatten()
        class_weight_temp = compute_class_weight(class_weight='balanced', classes=np.unique(y_integers), y=y_integers)
        class_weight_temp[1] *= 0.25
        print(class_weight_temp)

        # Create the model arch
        unet.summary(line_length=150)
        callback = k.callbacks.EarlyStopping(monitor='val_loss', patience=50, mode='min', restore_best_weights=True,
                                             start_from_epoch=5)  # Early stopping.

        # Compile model
        lr_schedule = k.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=0.1,
            decay_steps=50000,
            decay_rate=0.85,
            staircase=True)
        optimizer = k.optimizers.SGD(learning_rate=lr_schedule)
        unet.compile(loss=custom_weighted_binary_crossentropy(class_weight_temp[0], class_weight_temp[1]),
                     optimizer=optimizer,
                     metrics=[k.losses.Dice()])

        my_callbacks = [k.callbacks.ModelCheckpoint(filepath='./logs/model.{epoch:02d}-{val_loss:.2f}.keras'),
                        callback]
        unet.fit([self.x_train, self.x_train8], [self.y_train, self.y_train],
                 batch_size=self.batch_size,
                 epochs=self.epochs,
                 verbose=2,
                 validation_data=([self.x_val, self.x_val8], [self.y_val, self.y_val]),
                 callbacks=my_callbacks)
        return unet
