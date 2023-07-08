
from __future__ import print_function
from __future__ import absolute_import
import os
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import regularizers
from concactLayer import concatLayer
from mediumlayer import MediumLayer
from attention_layer import LayerNormalization, Position_Embedding, PositionWiseFeedForward, MultiHeadAttention
from MultiHeadAttention import MultiHeadAttention_
from LayerNormalization import LayerNormalization
from Position_Embedding import Position_Embedding
from PositionWiseFeedForward import PositionWiseFeedForward

tf.compat.v1.disable_eager_execution()

import pickle
import logging

logger = logging.getLogger(__name__)

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
        self.textEmbedding = pickle.load(open(self.dataParams['text_pretrain_emb_path'], "rb"), encoding='iso-8859-1')
        self.codeEmbedding = pickle.load(open(self.dataParams['code_pretrain_emb_path'], "rb"), encoding='iso-8859-1')

        if not os.path.exists(self.config['workdir'] + 'models/' + self.modelParams['model_name'] + '/'):
            os.makedirs(self.config['workdir'] + 'models/' + self.modelParams['model_name'] + '/')

        self.nb_classes = 2
        self.dropout1 = None
        self.dropout2 = None
        self.dropout3 = None
        self.dropout4 = None
        self.dropout5 = None
        self.regularizer = None
        self.random_seed = None
        self.num = None

    def paramsAdjust(self, dropout1=0.5, dropout2=0.5, dropout3=0.5, dropout4=0.5, dropout5=0.5, regularizer=0.01, num=100, seed=42):
        self.dropout1 = dropout1
        self.dropout2 = dropout2
        self.dropout3 = dropout3
        self.dropout4 = dropout4
        self.dropout5 = dropout5
        self.regularizer = regularizer
        self.random_seed = seed
        self.num = num

    def build(self):
        logger.debug('Building Code Representation Model')
        textS1 = Input(shape=(self.textLength,), dtype='int32', name='S1name')
        textS2 = Input(shape=(self.textLength,), dtype='int32', name='S2name')
        code = Input(shape=(self.codeLength,), dtype='int32', name='codename')
        queries = Input(shape=(self.queriesLength,), dtype='int32', name='queryname')

        embedding_layer = Embedding(self.textEmbedding.shape[0], self.textEmbedding.shape[1],
                                    weights=[self.textEmbedding], input_length=self.textLength,
                                    trainable=False, mask_zero=True)

        text_S1_embedding = embedding_layer(textS1)
        text_S2_embedding = embedding_layer(textS2)

        position_embedding = Position_Embedding(10, 'concat')
        text_S1_embedding_p = positionEmbedding(text_S1_embedding)
        text_S2_embedding_p = positionEmbedding(text_S2_embedding)

        dropout = Dropout(self.dropout1, name='dropout_embed', seed=self.random_seed)
        text_S1_embedding_d = dropout(text_S1_embedding_p)
        text_S2_embedding_d = dropout(text_S2_embedding_p)

        layer = MultiHeadAttention_(10)
        t1 = layer([text_S1_embedding_d, text_S1_embedding_d, text_S1_embedding_d])
        t2 = layer([text_S2_embedding_d, text_S2_embedding_d, text_S2_embedding_d])

        add_out = Lambda(lambda x: x[0] + x[1])
        t1 = add_out([t1, text_S1_embedding_d])
        t2 = add_out([t2, text_S2_embedding_d])

        t1_l = LayerNormalization()(t1)
        t2_l = LayerNormalization()(t2)

        ff =  PositionWiseFeedForward(310, 2048)
        ff_t1 = ff(t1_l)
        ff_t2 = ff(t2_l)

        dropout_ = Dropout(self.dropout2, name='dropout_ffn', seed=self.random_seed)
        ff_t1 = dropout_(ff_t1)
        ff_t2 = dropout_(ff_t2)

        ff_t1 = add_out([ff_t1, t1_l])
        ff_t2 = add_out([ff_t2, t2_l])

        t1 = LayerNormalization()(ff_t1)
        t2 = LayerNormalization()(ff_t2)

        dropout = Dropout(self.dropout3, name='dropout_qc', seed=self.random_seed)
        text_S1_semantic = GlobalAveragePooling1D(name='globaltext_1')(t1)
        text_S1_semantic = Activation(tf.nn.leaky_relu)(text_S1_semantic)
        text_S2_semantic = GlobalAveragePooling1D(name='globaltext_2')(t2)
        text_S2_semantic = Activation(tf.nn.leaky_relu)(text_S2_semantic)

        sentence_token_level_outputs = MediumLayer()([text_S1_semantic, text_S2_semantic])
        layer5 = Bidirectional(GRU(units=128, dropout=self.dropout4))
        f1 = layer5(sentence_token_level_outputs)
        dropout = Dropout(self.dropout5, name='dropout2', seed=self.random_seed)
        f1 = dropout(f1)

        classf = Dense(2, activation='softmax', name="final_class", kernel_regularizer=regularizers.l2(self.regularizer))(f1)

        class_model = Model(inputs=[text_S1, text_S2, code, queries], outputs=[classf], name='class_model')
        self.class_model = class_model

        print("\nSummary of class model:")
        self.class_model.summary()
        fname = self.config['workdir'] + 'models/' + self.modelParams['model_name'] + '/_class_model.png'
        P1, P2, Pc, Pq = None, None, None, None
        myloss = self.diceLoss(P1, P2, Pc, Pq)
        optimizer = Adam(learning_rate=0.001, clipnorm=0.001)
        self.class_model.compile(loss=myloss, optimizer=optimizer)

    def compile(self, optimizer, **kwargs):
        logger.info('Compiling models')
        self.class_model.compile(loss=self.exampleLoss, optimizer=optimizer, **kwargs)

    def fit(self, x, y, **kwargs):
        assert self.class_model is not None, 'Must compile the model before fitting data'
        return self.class_model.fit(x, to_categorical(y), **kwargs)

    def predict(self, x, **kwargs):
        return self.class_model.predict(x, **kwargs)

    def save(self, class_model_file, **kwargs):
        assert self.class_model is not None, 'Must compile the model before saving weights'
        self.class_model.save_weights(class_model_file, **kwargs)

    def load(self, class_model_file, **kwargs):
        assert self.class_model is not None, 'Must compile the model before loading weights'
        self.class_model.load_weights(class_model_file, **kwargs)

    def myCrossentropy(self, y_true, y_pred, e=0.1):
        loss1 = K.categorical_crossentropy(y_true, y_pred)
        loss2 = K.categorical_crossentropy(K.ones_like(y_pred) / self.nb_classes, y_pred)
        return (1 - e) * loss1 + e * loss2

    def exampleLoss(self, y_true, y_pred):
        crossent = tf.compat.v1.nn.softmax_cross_entropy_with_logits(logits=y_pred, labels=y_true)
        loss = tf.reduce_sum(crossent) / tf.cast(100, tf.float32)
        return loss

    def diceCoef(self, y_true, y_pred, p1, p2, p3, p4, e=0.1):
        loss1 = K.categorical_crossentropy(y_true, y_pred)
        loss2 = K.categorical_crossentropy(K.ones_like(y_pred) / self.nb_classes, y_pred)
        return (1 - e) * loss1 + e * loss2

    def diceLoss(self, p1, p2, p3, p4):
        def dice(y_true, y_pred):
            return self.diceCoef(y_true, y_pred, p1, p2, p3, p4)

        return dice
