import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, BatchNormalization, Activation

class Unet:
    def __init__(self, input_shape=(256, 256, 3), classes=1, activation='sigmoid', filters=64):
        self.input_shape = input_shape
        self.classes = classes
        self.activation = activation
        self.filters = filters

    def conv_block(self, inputs, filters):
        x = Conv2D(filters, 3, padding='same')(inputs)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2D(filters, 3, padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        return x

    def build(self):
        inputs = Input(self.input_shape)

        # encoder (downsampling)
        conv1 = self.conv_block(inputs, self.filters)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

        conv2 = self.conv_block(pool1, self.filters*2)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

        conv3 = self.conv_block(pool2, self.filters*4)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

        conv4 = self.conv_block(pool3, self.filters*8)
        pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

        # bottleneck
        conv5 = self.conv_block(pool4, self.filters*16)

        # decoder (upsampling)
        up6 = UpSampling2D(size=(2, 2))(conv5)
        up6 = concatenate([up6, conv4])
        conv6 = self.conv_block(up6, self.filters*8)

        up7 = UpSampling2D(size=(2, 2))(conv6)
        up7 = concatenate([up7, conv3])
        conv7 = self.conv_block(up7, self.filters*4)

        up8 = UpSampling2D(size=(2, 2))(conv7)
        up8 = concatenate([up8, conv2])
        conv8 = self.conv_block(up8, self.filters*2)

        up9 = UpSampling2D(size=(2, 2))(conv8)
        up9 = concatenate([up9, conv1])
        conv9 = self.conv_block(up9, self.filters)

        outputs = Conv2D(self.classes, 1, activation=self.activation)(conv9)

        model = Model(inputs=inputs, outputs=outputs)
        return model