#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <http://www.gnu.org/licenses/>.
#    
import tensorflow as tf
from keras.engine.topology import Layer
from keras import activations, initializers, regularizers, constraints

class TernaryLayer(Layer):
    """TernaryLayer:
    This layer applies the ternary tensor product
    y = xSW
    y = f(x * S * W + bs)

    # Notations:
    b: size of mini-batch
    n: number of input neurons
    m (or n_hidden): number of output neurons
    w (or n_weights): number of weights of the kernel
    p: number of input channels
    q (or output_dim): number of output feature maps

    f: activation function
    bs: bias

    # Tensors:
    x: layer input, of shape (b,n,p)
    S: (weight sharing) scheme tensor, of shape (w,n,m)
    kernel: weight tensor, of shape (w,p,q)
    y: layer output, of shape (b,m,q)
    """

    def __init__(self, n_weights, n_hidden, output_dim,

                 train_scheme=False, scheme_initializer='he_uniform', scheme_init_scale=1.0,
                 train_kernel=True, kernel_initializer='he_uniform', kernel_init_scale=1.0,
                 
                 activation=None, use_bias=True, bias_initializer='zeros',

                 scheme_constraint=None, kernel_constraint=None,
                 scheme_regularizer=None, kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,

                 **kwargs):

        super(TernaryLayer, self).__init__(**kwargs)

        self.w = n_weights
        self.q = output_dim
        self.m = n_hidden

        self.train_kernel = train_kernel
        self.train_scheme = train_scheme
        self.scheme_initializer = initializers.get(scheme_initializer)
        self.scheme_init_scale = scheme_init_scale
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.kernel_init_scale = kernel_init_scale

        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.bias_initializer = initializers.get(bias_initializer)

        self.scheme_constraint = scheme_constraint
        self.scheme_regularizer = regularizers.get(scheme_regularizer)
        self.kernel_constraint = kernel_constraint
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)

    def build(self, input_shape):
        assert len(input_shape) == 3
        self.n = input_shape[1]
        self.p = input_shape[2]
        self.init_scheme()
        self.init_kernel()
        self.init_biases()
        self.built = True

    def init_scheme(self):
        self.scheme = self.add_weight(shape=(self.w, self.n, self.m),
                            initializer= self.scheme_initializer,
                            name='{}_scheme'.format(self.name),
                            regularizer=self.scheme_regularizer,
                            constraint=self.scheme_constraint,
                            trainable=self.train_scheme)
        if self.scheme_init_scale != 1.0:
            self.scheme *= self.scheme_init_scale

    def init_kernel(self):
        self.kernel = self.add_weight(shape=(self.w, self.p, self.q),
                            initializer= self.kernel_initializer,
                            name='{}_kernel'.format(self.name),
                            regularizer=self.kernel_regularizer,
                            constraint=self.kernel_constraint,
                            trainable=self.train_kernel)
        if self.kernel_init_scale != 1.0:
            self.kernel *= self.kernel_init_scale

    def init_biases(self):
        if self.use_bias:
            self.bias = self.add_weight(shape=(self.q,),
                                        initializer=self.bias_initializer,
                                        name='{}_bias'.format(self.name),
                                        regularizer=self.bias_regularizer,
                                        constraint=None,
                                        trainable=self.train_kernel)
        else:
            self.bias = None

    def call(self, x, mask=None):

        # x (b,n,p) (dot) S (w,n,m) = xS (b,p,w,m)
        xS = tf.tensordot(x, self.scheme, [[1],[1]])

        # xS (b,p,w,m) (dot) W (w,p,q) = y (b,m,q)
        y = tf.tensordot(xS, self.kernel, [[1,2],[1,0]])
        y.set_shape([x.shape[0], self.m, self.q])

        if self.use_bias:
            y += tf.reshape(self.bias, (1, 1, self.q))
        return self.activation(y)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.m, self.q)
