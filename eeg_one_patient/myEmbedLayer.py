from keras import backend as K
from keras.engine.topology import Layer
import numpy as np

# Takes windows of size W (must be odd) of the input data supposed to enter in the form (num_samples, features).
# It outputs a matrix of size (num_samples, W, features)

class myEmbedLayer(Layer):

    def __init__(self, win_size=3, **kwargs):
        super(myEmbedLayer, self).__init__(**kwargs)
        self.win_size = win_size

    def build(self, input_shape):
        self.hw = (self.win_size - 1) / 2
        super(myEmbedLayer, self).build(input_shape)

    def compute_output_shape(self, input_shape):
        a = (input_shape[0],) +  (self.win_size, input_shape[1])
        return a

    def call(self, x, mask=None):
        #input_shape = self.input_spec[0].shape
        #input_shape = x.shape
        input_shape = K.int_shape(x)

        a = K.reshape(x, (-1,1,input_shape[1]))

        kernel = K.concatenate([K.constant(0, shape=(1, 0)), K.constant(1, shape=(1, 0))], axis = 0)
        b = K.conv1d(a, kernel,data_format = 'channel_last')

        #b = K.concatenate([a[1:,:,:],a[-1,1,:]],axis=0)
        y = K.concatenate([a,a,a],axis=1)
        return y

    def get_config(self):
        config = {'win_size': self.win_size}
        base_config = super(Cropping2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))