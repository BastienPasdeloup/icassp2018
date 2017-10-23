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
####################################################################################################################
################################################## INITIALIZATION ##################################################
####################################################################################################################

# Imports
import sys
import ast
import numpy
import tensorflow
from ternary_layer import *
from keras.models import Sequential
from keras.datasets import cifar10
from keras.layers import Reshape, Dense, Dropout, Flatten, Activation
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
from keras.utils.np_utils import to_categorical

# Arguments
if len(sys.argv) != 2 :
    print("Usage: " + sys.argv[0] + " <translationsFile>")
    quit()
translationsFile = sys.argv[1]

# Processing the translations file
with open(translationsFile, "r") as fileObject :
    fileContents = ast.literal_eval(fileObject.read())
nbVertices = len(fileContents)
maxFilterSize = max([value for value in max([list(cnnFilter.values()) for cnnFilter in fileContents])])

# Custom layer using these translations
S = numpy.zeros((maxFilterSize, nbVertices, nbVertices))
i = 0
for cnnFilter in fileContents:
    for vertex in cnnFilter :
        S[cnnFilter[vertex] - 1, vertex, i] = 1
    i += 1
initializer = tensorflow.constant_initializer(S)

# We load the dataset
(X_train, Y_train), (X_test, Y_test) = cifar10.load_data()
imgWidth = X_train.shape[1]
imgHeight = X_train.shape[2]
imgChannels = X_train.shape[3]
nbClasses = len(set([label[0] for label in Y_train]))

####################################################################################################################
################################################## CNN DESCRIPTION #################################################
####################################################################################################################

# Input of the model
model = Sequential()
model.add(Reshape((nbVertices, imgChannels), input_shape=(imgWidth, imgHeight, imgChannels)))

# Custom layer 1
nbFilters = 96
model.add(TernaryLayer(maxFilterSize, nbVertices, nbFilters, scheme_initializer=initializer, train_scheme=False))
model.add(BatchNormalization())
model.add(Activation("relu"))

# Custom layer 2
nbFilters = 96
model.add(TernaryLayer(maxFilterSize, nbVertices, nbFilters, scheme_initializer=initializer, train_scheme=False))
model.add(BatchNormalization())
model.add(Activation("relu"))

# Custom layer 3
nbFilters = 96
model.add(TernaryLayer(maxFilterSize, nbVertices, nbFilters, scheme_initializer=initializer, train_scheme=False))
model.add(BatchNormalization())
model.add(Activation("relu"))
model.add(Dropout(0.5))

# Custom layer 4
nbFilters = 192
model.add(TernaryLayer(maxFilterSize, nbVertices, nbFilters, scheme_initializer=initializer, train_scheme=False))
model.add(BatchNormalization())
model.add(Activation("relu"))

# Custom layer 5
nbFilters = 192
model.add(TernaryLayer(maxFilterSize, nbVertices, nbFilters, scheme_initializer=initializer, train_scheme=False))
model.add(BatchNormalization())
model.add(Activation("relu"))
model.add(Dropout(0.5))

# Custom layer 6
nbFilters = 192
model.add(TernaryLayer(maxFilterSize, nbVertices, nbFilters, scheme_initializer=initializer, train_scheme=False))
model.add(BatchNormalization())
model.add(Activation("relu"))
model.add(Dropout(0.5))

# Custom layer 7
nbFilters = 96
model.add(TernaryLayer(maxFilterSize, nbVertices, nbFilters, scheme_initializer=initializer, train_scheme=False))
model.add(BatchNormalization())
model.add(Activation("relu"))
model.add(Dropout(0.5))

# Dense output layer
model.add(Flatten())
model.add(Dense(nbClasses, activation="softmax"))

####################################################################################################################
################################################### CNN TEACHING ###################################################
####################################################################################################################

# First 150 epochs with learning rate of 0.001
model.compile(loss="categorical_crossentropy", optimizer=Adam(lr=0.001, decay=1e-6), metrics=["accuracy"])

model.summary()

model.fit(X_train / 255.0, to_categorical(Y_train), batch_size=32, shuffle=True, epochs=150, validation_data=(X_test / 255.0, to_categorical(Y_test)), verbose=1)

# Next 75 epochs with learning rate of 0.0001
model.compile(loss="categorical_crossentropy", optimizer=Adam(lr=0.0001, decay=1e-6), metrics=["accuracy"])
model.fit(X_train / 255.0, to_categorical(Y_train), batch_size=32, shuffle=True, epochs=75, validation_data=(X_test / 255.0, to_categorical(Y_test)), verbose=1)

# Final 75 epochs with learning rate of 0.00001
model.compile(loss="categorical_crossentropy", optimizer=Adam(lr=0.00001, decay=1e-6), metrics=["accuracy"])
model.fit(X_train / 255.0, to_categorical(Y_train), batch_size=32, shuffle=True, epochs=75, validation_data=(X_test / 255.0, to_categorical(Y_test)), verbose=1)
