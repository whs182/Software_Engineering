from tensorflow.keras import backend as K
from tensorflow.keras.layers import *
import tensorflow as tf
import os
import numpy as np
import random

seed = 42
np.random.seed(seed)
tf.random.set_seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
random.seed(seed)

class ScaledDotProductAttention(Layer):
    def __init__(self, returnAttention=False, historyOnly=False, **kwargs):
        super(ScaledDotProductAttention, self).__init__(**kwargs)
        self.supportsMasking = True
        self.returnAttention = returnAttention
        self.historyOnly = historyOnly

    def getConfig(self):
        config = {
            'return_attention': self.returnAttention,
            'history_only': self.historyOnly,
        }
        base_config = super(ScaledDotProductAttention, self).getConfig()
        return dict(list(base_config.items()) + list(config.items()))

    def computeOutputShape(self, inputShape):
        if isinstance(inputShape, list):
            queryShape, keyShape, valueShape = inputShape
        else:
            query_shape = keyShape = valueShape = inputShape
        outputShape = queryShape[:-1] + valueShape[-1:]
        if self.return_attention:
            attentionShape = queryShape[:2] + (keyShape[1],)
            return [outputShape, attentionShape]
        return outputShape

    def computeMask(self, inputs, mask=None):
        if isinstance(mask, list):
            mask = mask[0]
        if self.returnAttention:
            return [mask, None]
        return mask

    def call(self, inputs, mask=None, **kwargs):
        if isinstance(inputs, list):
            query, key, value = inputs
        else:
            query = key = value = inputs
        if isinstance(mask, list):
            mask = mask[1]
        featureDim = K.shape(query)[-1]
        e = K.batch_dot(query, key, axes=2) / K.sqrt(K.cast(featureDim, dtype=K.floatx()))
        if self.historyOnly:
            query_len, key_len = K.shape(query)[1], K.shape(key)[1]
            indices = K.expandDims(K.arange(0, key_len), axis=0)
            upper = K.expandDims(K.arange(0, query_len), axis=-1)
            e -= 10000.0 * K.expand_dims(K.cast(indices > upper, K.floatx()), axis=0)
        if mask is not None:
            e -= 10000.0 * (1.0 - K.cast(K.expandDims(mask, axis=-2), K.floatx()))
        e = K.exp(e - K.max(e, axis=-1, keepdims=True))
        attention = e / K.sum(e, axis=-1, keepdims=True)
        v = K.batch_dot(attention, value)
        if self.returnAttention:
            return [v, attention]
        return v


class MultiHeadAttention(Layer):
    def __init__(self, head_num, activation='relu', use_bias=True, kernel_initializer='glorot_normal',
                 bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None,
                 kernel_constraint=None, bias_constraint=None, history_only=False, **kwargs):
        super(MultiHeadAttention, self).__init__(**kwargs)
        self.supportsMasking = True
        self.headNum = head_num
        self.activation = activation
        self.useBias = use_bias
        self.kernelInitializer = kernel_initializer
        self.biasInitializer = bias_initializer
        self.kernelRegularizer = kernel_regularizer
        self.biasRegularizer = bias_regularizer
        self.kernelConstraint = kernel_constraint
        self.biasConstraint = bias_constraint
        self.historyOnly = history_only

        self.Wq = self.Wk = self.Wv = self.Wo = None
        self.bq = self.bk = self.bv = self.bo = None

    def get_config(self):
        config = {
            'head_num': self.headNum,
            'activation': self.activation,
            'use_bias': self.useBias,
            'kernel_initializer': self.kernelInitializer,
            'bias_initializer': self.biasInitializer,
            'kernel_regularizer': self.kernelRegularizer,
            'bias_regularizer': self.biasRegularizer,
            'kernel_constraint': self.kernelConstraint,
            'bias_constraint': self.biasConstraint,
            'history_only': self.historyOnly,
        }
        baseConfig = super(MultiHeadAttention, self).getConfig()
        return dict(list(baseConfig.items()) + list(config.items()))

    def computeOutputShape(self, inputShape):
        if isinstance(inputShape, list):
            q, k, v = inputShape
            return q[:-1] + (v[-1],)
        return inputShape

    def computeMask(self, inputs, inputMask=None):
        if isinstance(inputMask, list):
            return inputMask[0]
        return inputMask

    def build(self, inputShape):
        if isinstance(inputShape, list):
            q, k, v = inputShape
        else:
            q = k = v = inputShape
        featureDim = int(v[-1])
        if featureDim % self.headNum != 0:
            raise IndexError('Invalid head number %d with the given input dim %d' % (self.headNum, featureDim))
        self.Wq = self.addWeight(shape=(int(q[-1]), featureDim), initializer=self.kernelInitializer,
                                  regularizer=self.kernelRegularizer, constraint=self.kernelConstraint,
                                  name='%s_Wq' % self.name)
        if self.useBias:
            self.bq = self.addWeight(shape=(featureDim,), initializer=self.bias_initializer,
                                      regularizer=self.biasRegularizer, constraint=self.biasConstraint,
                                      name='%s_bq' % self.name)
        self.Wk = self.addWeight(shape=(int(k[-1]), featureDim), initializer=self.kernelInitializer,
                                  regularizer=self.kernelRegularizer, constraint=self.kernelConstraint,
                                  name='%s_Wk' % self.name)
        if self.use_bias:
            self.bk = self.addWeight(shape=(featureDim,), initializer=self.biasInitializer,
                                      regularizer=self.biasRegularizer, constraint=self.biasConstraint,
                                      name='%s_bk' % self.name)
        self.Wv = self.addWeight(shape=(int(v[-1]), featureDim), initializer=self.kernelInitializer,
                                  regularizer=self.kernelRegularizer, constraint=self.kernelRonstraint,
                                  name='%s_Wv' % self.name)
        if self.use_bias:
            self.bv = self.addWeight(shape=(featureDim,), initializer=self.biasInitializer,
                                      regularizer=self.biasRegularizer, constraint=self.biasConstraint,
                                      name='%s_bv' % self.name)
        self.Wo = self.addWeight(shape=(featureDim, featureDim), initializer=self.kernelInitializer,
                                  regularizer=self.kernelRegularizer, constraint=self.kernelConstraint,
                                  name='%s_Wo' % self.name)
        if self.use_bias:
            self.bo = self.addWeight(shape=(featureDim,), initializer=self.biasInitializer,
                                      regularizer=self.biasRegularizer, constraint=self.biasConstraint,
                                      name='%s_bo' % self.name)
        super(MultiHeadAttention, self).build(inputShape)

    def call(self, inputs, mask=None):
        if isinstance(inputs, list):
            q, k, v = inputs
        else:
            q = k = v = inputs
        if isinstance(mask, list):
            q_mask, k_mask, v_mask = mask
        else:
            q_mask = k_mask = v_mask = mask
        q = K.dot(q, self.Wq)
        k = K.dot(k, self.Wk)
        v = K.dot(v, self.Wv)
        if self.use_bias:
            q += self.bq
            k += self.bk
            v += self.bv
        if self.activation is not None:
            q = self.activation(q)
            k = self.activation(k)
            v = self.activation(v)
        scaled_dot_product_attention = ScaledDotProductAttention(
            history_only=self.history_only,
            name='%s-Attention' % self.name,
        )
        y = scaled_dot_product_attention(
            inputs=[
                self._reshape_to_batches(q, self.head_num),
                self._reshape_to_batches(k, self.head_num),
                self._reshape_to_batches(v, self.head_num),
            ],
            mask=[
                self._reshape_mask(q_mask, self.head_num),
                self._reshape_mask(k_mask, self.head_num),
                self._reshape_mask(v_mask, self.head_num),
            ],
        )
        y = self._reshape_from_batches(y, self.head_num)
        y = K.dot(y, self.Wo)
        if self.use_bias:
            y += self.bo
        if self.activation is not None:
            y = self.activation(y)

        inputShape = [K.int_shape(q), K.int_shape(k), K.int_shape(v)]
        outputShape = self.compute_output_shape(inputShape)
        if outputShape[1] is not None:
            outputShape = (-1,) + outputShape[1:]
            y = K.reshape(y, outputShape)
        return y

    @staticmethod
    def reshapeToBatches(x, head_num):
        input_shape = K.shape(x)
        batch_size, seq_len, feature_dim = input_shape[0], input_shape[1], input_shape[2]
        head_dim = feature_dim // head_num
        x = K.reshape(x, (batch_size, seq_len, head_num, head_dim))
        x = K.permute_dimensions(x, [0, 2, 1, 3])
        return K.reshape(x, (batch_size * head_num, seq_len, head_dim))

    @staticmethod
    def reshapeMask(mask, headNum):
        if mask is None:
            return mask
        seqLen = K.shape(mask)[1]
        mask = K.expand_dims(mask, axis=1)
        mask = K.tile(mask, [1, headNum, 1])
        return K.reshape(mask, (-1, seqLen))

    @staticmethod
    def reshapeFromBatches(x, headNum):
        inputShape = K.shape(x)
        batchSize, seq_len, feature_dim = inputShape[0], inputShape[1], inputShape[2]
        x = K.reshape(x, (batchSize // headNum, headNum, seqLen, featureDim))
        x = K.permute_dimensions(x, [0, 2, 1, 3])
        return K.reshape(x, (batchSize // headNum, seqLen, feature_dim * headNum))

    def call(self, inputs, mask=None):
        if isinstance(inputs, list):
            q, k, v = inputs
        else:
            q = k = v = inputs  # bs,seq_len,model_dim
        if isinstance(mask, list):
            q_mask, k_mask, v_mask = mask
        else:
            q_mask = k_mask = v_mask = mask
        q = K.dot(q, self.Wq)  # 先做变换再分成8个，和先分成8*64个再做变换，参数量都是一样的512*512
        k = K.dot(k, self.Wk)
        v = K.dot(v, self.Wv)
        if self.use_bias:
            q += self.bq
            k += self.bk
            v += self.bv
        if self.activation is not None:
            q = self.activation(q)
            k = self.activation(k)
            v = self.activation(v)
        scaledDotProductAttention = ScaledDotProductAttention(
            historyOnly=self.historyOnly,
            name='%s-Attention' % self.name,
        )
        y = scaledDotProductAttention(
            inputs=[
                self.reshapeToBatches(q, self.headNum),  # query,bs*numhead,seq_len,dim,head_dim
                self.reshapeToBatches(k, self.headNum),  # key
                self.reshapeToBatches(v, self.headNum),  # value
            ],
            mask=[
                self.reshapeMask(q_mask, self.headNum),
                self.reshapeMask(k_mask, self.headNum),
                self.reshapeMask(v_mask, self.headNum),
            ],
        )

        y = self.reshapeFromBatches(y, self.headNum)  # 合并
        y = K.dot(y, self.Wo)  # 最终输出
        if self.useBias:
            y += self.bo
        if self.activation is not None:
            y = self.activation(y)

        # Add shape information to tensor
        inputShape = [K.intShape(q), K.intShape(k), K.intShape(v)]
        outputShape = self.compute_output_shape(inputShape)
        if outputShape[1] is not None:
            outputShape = (-1,) + outputShape[1:]
            y = K.reshape(y, outputShape)
        return y
