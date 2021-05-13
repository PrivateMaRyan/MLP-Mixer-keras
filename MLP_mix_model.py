import keras 
import math
import numpy as np
from keras.layers import *
import keras.backend as K
import tensorflow as tf
from keras.models import Model
from keras_layer_normalization import LayerNormalization
from keras.utils import get_custom_objects

def gelu(x):
    cdf = 0.5 * (1.0 + tf.tanh(
      (np.sqrt(2 / np.pi) * (x + 0.044715 * tf.pow(x, 3)))))
    return x * cdf

get_custom_objects().update({"gelu":gelu})

class ChannelMixBlock():
    def __init__(self, name, expand_ratio = 2):
        self.name = "chnmix_{}".format(name)
        self.expand_ratio = expand_ratio
    def __call__(self, x):
        _, h, w, c = K.int_shape(x)
        y = x
        x = LayerNormalization()(x)
        x = Conv2D(c * self.expand_ratio, (1, 1), name = self.name + "_fc1")(x)
        x = Activation("gelu", name = self.name + "_activation")(x)
        x = Conv2D(c, (1, 1), name = self.name + "_fc2")(x)
        x = Add(name = self.name + "_add")([x, y])
        return x
    
class TokenMixBlock():
    def __init__(self, name, expand_ratio = 2):
        self.name = "tkmix_{}".format(name)
        self.expand_ratio = expand_ratio
        
    def __call__(self, x):
        _, h, w, c = K.int_shape(x)
        new_channel = int(math.sqrt(c))
        y = x
        x = LayerNormalization()(x)
        x = Reshape((h * w, c), name = self.name + "_reshape1")(x)
        x = Permute([1, 0], name = self.name + "_permute1")(x)
        x = Reshape((new_channel, new_channel, h*w), name = self.name + "_reshape2")(x)
        x = Conv2D(h * w * self.expand_ratio, (1, 1), name = self.name + "_fc1")(x)
        x = Activation("gelu", name = self.name + "_activation")(x)
        x = Conv2D(h * w, (1, 1), name = self.name + "_fc2")(x)
        x = Reshape((c, h *w ), name = self.name + "_reshape3")(x)
        x = Permute([1, 0], name = self.name + "_permute2")(x)
        x = Reshape((h, w, c), name = self.name + "_reshape4")(x)
        x = Add(name = self.name + "_add")([x, y])
        return x

class MLPLayer():
    def __init__(self, name, expand_ratio):
        self.name = name
        self.expand_ratio = expand_ratio
        
    def __call__(self, x):
        x = ChannelMixBlock(self.name, self.expand_ratio)(x)
        x = TokenMixBlock(self.name, self.expand_ratio)(x)
        return x
    
def built_mlp_model(input_shape, mlp_depth, expand_ratio, patch_size, channels, num_classes = 1000):
    input_tensor = Input(input_shape)
    img_h = input_shape[0]
    assert (img_h % patch_size == 0), "input image size must be devided evenly by patch_size!"
    assert (math.sqrt(channel) %1 == 0), "channels must be perfect square!"

    x = Conv2D(channels, kernel_size = (patch_size, patch_size), strides= (patch_size, patch_size), name = "pre-patches")(input_tensor)
    for i in range(mlp_depth):
        x = MLPLayer(i, expand_ratio)(x)
    x = GlobalAveragePooling2D()(x)
    x = Dense(num_classes, activation = "softmax")(x)
    model = Model(input_tensor, x, name = "MLP-model")
    return model
