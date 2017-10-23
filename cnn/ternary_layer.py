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
    y = g(x * f(S * W + be) + bn)

    # Notations:
    b: size of mini-batch
    n: number of input neurons
    m (or n_hidden): number of output neurons
    w (or n_weights): number of weights of the kernel
    p: number of input channels
    q (or output_dim): number of output feature maps

    f: edge activation function
    g: neuron activation function
    be: edge bias
    bn: neuron bias

    # Tensors:
    x: layer input, of shape (b,n,p)
    S: scheme tensor, of shape (w,n,m)
    kernel: weights tensor, of shape (w,p,q)
    y: layer output, of shape (b,m,q)
    """

    def __init__(self, n_weights, n_hidden, output_dim,

                 train_scheme=True,scheme_initializer='he_uniform', scheme_init_scale=1.0,
                 train_kernel= True, kernel_initializer='he_uniform', kernel_init_scale=1.0,
                 
                 neuron_activation=None, use_neuron_bias=True, neuron_bias_initializer='zeros',
                 edge_activation=None, use_edge_bias=False, edge_bias_initializer='zeros',

                 scheme_constraint=None, kernel_constraint=None,
                 scheme_regularizer=None, kernel_regularizer=None,
                 neuron_bias_regularizer=None, edge_bias_regularizer=None,
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

        self.neuron_activation = activations.get(neuron_activation)
        self.edge_activation = activations.get(edge_activation)
        self.use_neuron_bias = use_neuron_bias
        self.use_edge_bias = use_edge_bias
        self.neuron_bias_initializer = initializers.get(neuron_bias_initializer)
        self.edge_bias_initializer = initializers.get(edge_bias_initializer)

        self.scheme_constraint = scheme_constraint
        self.scheme_regularizer = regularizers.get(scheme_regularizer)
        self.kernel_constraint = kernel_constraint
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.neuron_bias_regularizer = regularizers.get(neuron_bias_regularizer)
        self.edge_bias_regularizer = regularizers.get(edge_bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)

        self.use_faster_call = not(use_edge_bias) and edge_activation is None

    def build(self, input_shape):
        assert len(input_shape) == 3
        self.n = input_shape[1]
        self.p = input_shape[2]
        #self.check_requirements()
        self.init_scheme()
        self.init_kernel()
        self.init_biases()
        self.built = True

    def check_requirements(self):
        if 1. / self.w < 1. / (float(self.p) * self.q) + 1. / (float(self.n) * self.m):
            raise RuntimeError(
                'Dimensions of ternary layer ' + self.name + ' does not meet weight sharing requirements')

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
        if self.use_neuron_bias:
            self.neuron_bias = self.add_weight(shape=(self.q,),
                                        initializer=self.neuron_bias_initializer,
                                        name='{}_neuron_bias'.format(self.name),
                                        regularizer=self.neuron_bias_regularizer,
                                        constraint=None,
                                        trainable=self.train_kernel)
        else:
            self.neuron_bias = None

        if self.use_edge_bias:
            self.edge_bias = self.add_weight(shape=(self.m,),
                                        initializer=self.edge_bias_initializer,
                                        name='{}_edge_bias'.format(self.name),
                                        regularizer=self.edge_bias_regularizer,
                                        constraint=None,
                                        trainable=self.train_scheme)
        else:
            self.edge_bias = None

    def call(self, x, mask=None):
        if self.use_faster_call:
            return self.faster_call(x)

        # S (w,n,m) (dot) W (w,p,q) = SW (n,m,p,q)
        SW = tf.tensordot(self.scheme, self.kernel, [[0],[0]])
        if self.use_edge_bias:
            SW += tf.reshape(self.edge_bias, (1, self.m, 1, 1))
        SW = self.edge_activation(SW)

        # x (b,n,p) (dot) SW (n,m,p,q) = y (b,m,q)
        y = tf.tensordot(x, SW, [[1,2],[0,2]])
        y.set_shape([x.shape[0], self.m, self.q])
        if self.use_neuron_bias:
            y += tf.reshape(self.neuron_bias, (1, 1, self.q))
        return self.neuron_activation(y)

    def faster_call(self, x, mask=None):
        """This is used when there is no edge activation nor edge bias"""

        # x (b,n,p) (dot) S (w,n,m) = xS (b,p,w,m)
        xS = tf.tensordot(x, self.scheme, [[1],[1]])

        # xS (b,p,w,m) (dot) W (w,p,q) = y (b,m,q)
        y = tf.tensordot(xS, self.kernel, [[1,2],[1,0]])
        y.set_shape([x.shape[0], self.m, self.q])

        if self.use_neuron_bias:
            y += tf.reshape(self.neuron_bias, (1, 1, self.q))
        return self.neuron_activation(y)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.m, self.q)
