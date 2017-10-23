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

# Constants
NB_SESSIONS = 12
NB_INITIALIZATIONS = 25

# Arguments
if len(sys.argv) != 3 :
    print("Usage: " + sys.argv[0] + " <translationsFile> <subjectDirectory>")
    quit()
TRANSLATIONS_FILE = sys.argv[1]
SUBJECT_DIRECTORY = sys.argv[2]

# Processing the translations file
with open(TRANSLATIONS_FILE, "r") as fileObject :
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

####################################################################################################################
######################################################## CNN #######################################################
####################################################################################################################

# Functions to load the dataset
def loadData (sessionNumber) :
    with open(SUBJECT_DIRECTORY + "/session_" + str(sessionNumber) + "/fmri.data", "r") as fileObject :
        return numpy.asarray(ast.literal_eval(fileObject.read()))
def loadLabels (sessionNumber) :
    with open(SUBJECT_DIRECTORY + "/session_" + str(sessionNumber) + "/y.data", "r") as fileObject :
        mapping = {"cat": 0, "scrambledpix": 1, "chair": 2, "bottle": 3, "shoe": 4, "house": 5, "face": 6, "scissors": 7}
        strings = ast.literal_eval(fileObject.read())
        ints = [[mapping[label[0]]] for label in strings]
        return numpy.asarray(ints)
    
# Training data
X_train = loadData(0)
Y_train = loadLabels(1)
for session in [1, 2, 3, 4, 5, 6, 7, 8, 9] :
    X_train = numpy.concatenate((X_train, loadData(session)))
    Y_train = numpy.concatenate((Y_train, loadLabels(session)))

# Test data
X_test = loadData(10)
Y_test = loadLabels(10)
X_test = numpy.concatenate((X_test, loadData(11)))
Y_test = numpy.concatenate((Y_test, loadLabels(11)))

# Dimensions
nbVertices = X_train.shape[1]
nbClasses = len(set([label[0] for label in Y_train]))

# Grid search
gridSearchMax = 0.0
for cnnLayerNbFeatureMaps in [1, 2, 3] :
    for fullyConnectedSize1 in [30, 50, 100] :
        for fullyConnectedSize2 in [30, 50, 100] :
            for dropout in [0.2, 0.3, 0.5] :

                # We keep the best max found for various initializations
                for init in range(NB_INITIALIZATIONS) :
                
                    # Model
                    model = Sequential()
                    model.add(TernaryLayer(maxFilterSize, nbVertices, cnnLayerNbFeatureMaps, scheme_initializer=initializer, train_scheme=False, input_shape=(nbVertices,1)))
                    model.add(BatchNormalization())
                    model.add(Activation("relu"))
                    model.add(Flatten())
                    model.add(Dense(fullyConnectedSize1, activation="relu"))
                    model.add(Dense(fullyConnectedSize2, activation="relu"))
                    model.add(Dropout(dropout))
                    model.add(Dense(nbClasses, activation="softmax"))

                    # Training
                    model.compile(loss="categorical_crossentropy", optimizer=Adam(lr=0.001, decay=1e-6), metrics=["accuracy"])
                    model.summary()
                    history = model.fit(X_train, to_categorical(Y_train), batch_size=32, shuffle=True, epochs=50, validation_data=(X_test, to_categorical(Y_test)), verbose=0)
                    
                    # We save the max performance for every configuration
                    maxValue = max(history.history["val_acc"])
                    if maxValue > gridSearchMax :
                        gridSearchMax = maxValue
                        print("Current best: " + str(gridSearchMax) + " (cnnLayerNbFeatureMaps=" + str(cnnLayerNbFeatureMaps) + ", fullyConnectedSize1=" + str(fullyConnectedSize1) + ", fullyConnectedSize2=" + str(fullyConnectedSize2) + ", dropout=" + str(dropout) + ")")

# We save the max result for this training/test repartition
print("Best found: " + str(gridSearchMax))
