from __future__ import print_function
from __future__ import absolute_import
import os
from tensorflow.keras import Input
from tensorflow.keras.layers import *
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
import numpy as np
import pickle
import logging
from tensorflow.keras.optimizers import RMSprop, Adam, SGD

from tensorflow.keras.utils import *
from mediumlayer import *
import random

import tensorflow as tf
seed = 1
tf.compat.v1.set_random_seed(seed) #图级种子，使所有操作会话生成的随机序列在会话中可重复，请设置图级种子：
random.seed(seed)#让每次生成的随机数一致
np.random.seed(seed)
logger = logging.getLogger(__name__)

'''
biv-hnn模型
'''
class CodeMF:
    def __init__(self, config):
        self.config = config
        self.textLength = 100
        self.queriesLength = 25
        self.codeLength = 350
        self.classModel = None
        self.trainModel = None
        self.textS1 = Input(shape=(self.textLength,), dtype='int32', name='i_S1name')
        self.textS2 = Input(shape=(self.textLength,), dtype='int32', name='i_S2name')
        self.code = Input(shape=(self.codeLength,), dtype='int32', name='i_codename')
        self.queries = Input(shape=(self.queriesLength,), dtype='int32', name='i_queryname')
        self.labels = Input(shape=(1,), dtype='int32', name='i_queryname')
        self.nb_classes = 2
        self.dropout = None

        self.modelParams = config.get('model_params', dict())
        self.dataParams = config.get('data_params', dict())
        self.textEmbbeding = pickle.load(open(self.dataParams['text_pretrain_emb_path'], "rb"), encoding='iso-8859-1')
        self.codeEmbbeding = pickle.load(open(self.dataParams['code_pretrain_emb_path'], "rb"), encoding='iso-8859-1')
        # create a model path to store model info
        if not os.path.exists(self.config['workdir'] + 'models/' + self.modelParams['model_name'] + '/'):
            os.makedirs(self.config['workdir'] + 'models/' + self.modelParams['model_name'] + '/')

    def paramsAdjust(self, dropout1=0.5, dropout2=0.5, dropout3=0.5, dropout4=0.5, dropout5=0.5, Regularizer=0.01,
                      num=100,
                      seed=42):
        self.dropout = dropout1
        self.dropout2 = dropout2
        self.dropout3 = dropout3
        self.dropout4 = dropout4
        self.dropout5 = dropout5
        self.Regularizer = Regularizer
        self.random_seed = seed

        self.num = num

    def build(self):
        '''
        1. Build Code Representation Model
        '''
        logger.debug('Building Code Representation Model')
        textS1 = Input(shape=(self.textLength,), dtype='int32', name='S1name')
        textS2 = Input(shape=(self.textLength,), dtype='int32', name='S2name')
        code = Input(shape=(self.codeLength,), dtype='int32', name='codename')
        queries = Input(shape=(self.queriesLength,), dtype='int32', name='queryname')

        '''
        2.Embedding
        '''
        embeddingLayer = Embedding(self.textEmbbeding.shape[0], self.textEmbbeding.shape[1],
                                    weights=[self.textEmbbeding], input_length=self.textLength,
                                    trainable=False, mask_zero=True)

        text_S1_embeding = embeddingLayer(textS1)
        text_S2_embeding = embeddingLayer(textS2)
        emneddingLayer = Embedding(self.textEmbbeding.shape[0], self.textEmbbeding.shape[1],
                                    weights=[self.textEmbbeding], input_length=self.queriesLength,
                                    trainable=False, mask_zero=True)

        queriesEmbeding = emneddingLayer(queries)

        embeddingLayer = Embedding(self.codeEmbbeding.shape[0], self.codeEmbbeding.shape[1],
                                    weights=[self.codeEmbbeding], input_length=self.codeLength,
                                    trainable=False, mask_zero=True)
        codeEmbeding = embeddingLayer(code)
        dropout = Dropout(self.dropout, name='dropout_embed',seed = 1)
        text_S1_embeding = dropout(text_S1_embeding)
        text_S2_embeding = dropout(text_S2_embeding)
        code_embeding = dropout(codeEmbeding)
        queries_embeding = dropout(queriesEmbeding)

        '''
        3. 双向gru
       '''

        layer = Bidirectional(GRU(units=64))
        t1 = layer(text_S1_embeding)
        t2 = layer(text_S2_embeding)
        c = layer(code_embeding)
        q = layer(queries_embeding)


        '''
        query and code
        '''
        layer2 = Lambda(lambda x: tf.concat([x[0], x[1]], axis=1))
        code_q = layer2([c, q])
        code_q = dropout(code_q)
        layer2 = Dense(128, activation='tanh')
        file_c = layer2(code_q)
        sentence = MediumLayer()(
            [t1, t2, file_c])

        sentence = dropout(sentence)

        layer5 = Bidirectional(GRU(units=128, return_sequences=True))

        f1 = layer5(sentence)
        f1 = Lambda(lambda x: K.permute_dimensions(x, (1, 0, 2)))(f1)

        f1 = Lambda(lambda x: tf.unstack(x, axis=0))(f1)
        f1 = Lambda(lambda x: x[1])(f1)
        f1 = dropout(f1)
        '''
        sentence = SeqSelfAttention(1, 3)(sentence)
        sentence = dropout(sentence)
        '''
        classf = Dense(2, activation='softmax', name="final_class")(f1)

        class_model = Model(inputs=[textS1, textS2, code, queries], outputs=[classf], name='class_model')
        self.class_model = class_model

        print("\nsummary of class model")
        self.class_model.summary()
        fname = self.config['workdir'] + 'models/' + self.modelParams['model_name'] + '/_class_model.png'

        '''
        7.train model
        '''
       
        pred = class_model([self.text_S1,self.text_S2,self.code,self.queries])
        loss = Lambda(lambda x:K.minimum(1e-6,K.categorical_crossentropy(self.labels,x)+0.2),output_shape= lambda x:(1,),name='newloss')(pred)
        self.train_model = Model(inputs=[self.text_S1,self.text_S2,self.code,self.queries,self.labels],outputs=[loss],name='train_model')
        self.train_model.summary()
        

        optimizer = Adam(learning_rate=0.001, clipnorm=0.001)
        self.class_model.compile(loss='categorical_crossentropy', optimizer=optimizer)

    def compile(self, optimizer, **kwargs):
        logger.info('compiling models')
        '''
        model_dice = self.dice_loss(smooth=1e-5, thresh=0.5)
        model.compile(loss=model_dice)
        '''
        self.class_model.compile(loss=self.example_loss, optimizer=optimizer, **kwargs)

    def fit(self, x, y, **kwargs):
        assert self.class_model is not None, 'Must compile the model before fitting data'
        return self.class_model.fit(x, to_categorical(y), **kwargs)
        '''
        assert self.train_model is not None, 'Must compile the model before fitting data'
        y = np.zeros(shape=x[0].shape[:1], dtype=np.float32)
        return self.train_model.fit(x, y, **kwargs)
        '''

    def predict(self, x, **kwargs):
        return self.class_model.predict(x, **kwargs)

    def save(self, class_model_file, **kwargs):
        assert self.class_model is not None, 'Must compile the model before saving weights'
        self.class_model.save_weights(class_model_file, **kwargs)

    def load(self, class_model_file, **kwargs):
        assert self.class_model is not None, 'Must compile the model loading weights'
        self.class_model.load_weights(class_model_file, **kwargs)

    def concat(self, inputs):
        block_level_code_output = tf.split(inputs, inputs.shape[1], axis=1)
        block_level_code_output = tf.concat(block_level_code_output, axis=2)
        # (bs,600)
        block_level_code_output = tf.squeeze(block_level_code_output, axis=1)
        return block_level_code_output

    def mycrossentropy(self, y_true, y_pred, e=0.1):
        loss1 = K.categorical_crossentropy(y_true, y_pred)
        loss2 = K.categorical_crossentropy(K.ones_like(y_pred) / self.nb_classes, y_pred)
        return (1 - e) * loss1 + e * loss2

    def dice_coed_test(self, y_true, y_pred, P):
        e = 0.1
        loss1 = K.categorical_crossentropy(y_true, y_pred)
        loss2 = K.categorical_crossentropy(K.ones_like(y_pred) / self.nb_classes, y_pred)

        total_loss = (1 - e) * loss1 + e * loss2
        print(totalLoss)
        print(P)
        totalLoss = totalLoss
        return totalLoss

    def diceLossTest(self, P):
        def diceTest(y_true, y_pred):
            return self.dice_coed_test(y_true, y_pred, P)

        return diceTest

    def exampleLoss(self, y_true, y_pred):
        crossent = tf.compat.v1.nn.softmax_cross_entropy_with_logits(logits=y_pred, labels=y_true)
        # crossent = K.categorical_crossentropy(y_true, y_pred)
        loss = tf.reduce_sum(crossent) / tf.cast(100, tf.float32)

        return loss


'''
    def dice_coef(self,y_true, y_pred, smooth, thresh):
        y_true_f = K.flatten(y_true)
        y_pred_f = K.flatten(y_pred)
        intersection = K.sum(tf.multiply(y_true_f,y_pred_f))
        return (2* intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f)+smooth)

    def dice_loss(self,smooth, thresh):
        def dice(y_true, y_pred):
            return -self.dice_coef(y_true, y_pred, smooth, thresh)

        return dice
'''
