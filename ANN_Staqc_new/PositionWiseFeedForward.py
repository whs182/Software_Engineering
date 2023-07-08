from __future__ import print_function
from tensorflow.keras import backend as K
from tensorflow.keras.layers import *
import tensorflow as tf
from tensorflow import *
import os
import numpy as np
import random
import tensorflow as tf
seed = 42
np.random.seed(seed)
tf.random.set_seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
random.seed(seed)
class PositionWiseFeedForward(Layer):
    #inner_dim 隐藏层的维度，一般默认2048,model_dim是词向量的维度
    def __init__(self, model_dim, inner_dim, trainable=True, **kwargs):
        self._model_dim = model_dim
        self._inner_dim = inner_dim
        self._trainable = trainable
        super(PositionWiseFeedForward, self).__init__(**kwargs)

    def build(self, input_shape):
        self.weightsInner = self.add_weight(
            shape=(input_shape[-1], self._inner_dim),
            initializer='glorot_uniform',
            trainable=self._trainable,
            name="weights_inner")
        self.weightsOut = self.addWeight(
            shape=(self._inner_dim, self.modelDim),
            initializer='glorot_uniform',
            trainable=self._trainable,
            name="weights_out")
        self.baisInner = self.addWeight(
            shape=(self._inner_dim,),
            initializer='uniform',
            trainable=self._trainable,
            name="bais_inner")
        self.baisOut = self.addWeight(
            shape=(self._modelDim,),
            initializer='uniform',
            trainable=self._trainable,
            name="bais_out")
        super(PositionWiseFeedForward, self).build(input_shape)

    def call(self, inputs):
        if K.dtype(inputs) != 'float32':
            inputs = K.cast(inputs, 'float32')
        inner_out = K.relu(K.dot(inputs, self.weights_inner) + self.bais_inner)
        outputs = K.dot(inner_out, self.weights_out) + self.bais_out
        print("==",outputs.shape)
        return outputs

    def computeOutputShape(self, inputShape):
        return inputShape
'''
query = tf.random.truncated_normal([100, 50, 150])
w = PositionWiseFeedForward(150,2048)(query)
print(w.shape)
'''
