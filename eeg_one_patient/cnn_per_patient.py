'''Trains a CNN over the time segments of data
'''
from __future__ import print_function
from keras import regularizers
from keras.layers import Bidirectional
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, BatchNormalization, Flatten
from keras.layers.convolutional import Conv1D, MaxPooling1D, Conv2D, MaxPooling2D
from keras.optimizers import SGD
from os import listdir
from scipy.io import loadmat
from scipy.io import savemat
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
from keras import backend as K
from keras import regularizers
#import matplotlib.pyplot as plt
import numpy as np
np.random.seed(7)
import math
import pickle
from keras import metrics
from keras.callbacks import EarlyStopping

from keras.layers.wrappers import TimeDistributed
from keras.layers import LSTM
#import myEmbedLayer
import random
random.seed(45)
import os
import tensorflow as tf
tf.set_random_seed(45)

def comp_metric(y_true, y_pred):
    fp = sum(np.logical_and(y_true == 0, y_pred == 1))
    fn = sum(np.logical_and(y_true == 1, y_pred == 0))
    tp = sum(np.logical_and(y_true == 1, y_pred == 1))
    tn = sum(np.logical_and(y_true == 0, y_pred == 0))
    sensitivity = 100.0*float(tp) /float((tp + fn))
    return sensitivity, fp

def balance_dataset(X,Y):
    num_pos = sum(Y == 1)
    num_neg = sum(Y == 0)
    X_positive = X[(Y == 1).reshape(X.shape[0]),:]
    X_negative = X[(Y == 0).reshape(X.shape[0]),:]
    Y_positive = Y[(Y == 1).reshape(X.shape[0]),:]
    Y_negative = Y[(Y == 0).reshape(X.shape[0]),:]
    ran_vec=np.random.randint(low=0, high=num_neg, size=num_pos)
    X_negative = X_negative[ran_vec,:]
    Y_negative = Y_negative[ran_vec, :]
    X = np.concatenate((X_positive,X_negative))
    Y = np.concatenate((Y_positive, Y_negative))
    #X_ag = np.concatenate((X,Y),axis=1)
    #X_sh = np.random.shuffle(X_ag)
    return X,Y

def data_generator_one_patient(main_folder, patient_number,size_in, leaveout_sample, isTrain=True):
    nb_classes = 2
    patient_folder = main_folder + 'chb' + str(patient_number).zfill(2)
    print(patient_folder)
    list_samples = listdir(patient_folder)
    print(list_samples)
    if (isTrain):
        # take all series except for the one for testing
        X = np.zeros(shape=(0, size_in, 23,1))
        Y = np.zeros(shape=(0, 1))
        for sample in list_samples:
            # if is not the one to be tested
            if (sample != ('chb' + str(patient_number).zfill(2) + '_' + str(leaveout_sample).zfill(2))):
                print(sample)
                mat_var = loadmat(main_folder + 'chb' + str(patient_number).zfill(2) + '/' + sample)
                X_train = mat_var['total_images']
                X_train = X_train.reshape(X_train.shape[0],size_in,  23,1)
                Y_train = mat_var['total_labels']
                X_train = np.compress((Y_train != 2).flatten(), X_train, 0)  # get rid of pre-ictal
                Y_train = np.compress((Y_train != 2).flatten(), Y_train, 0)  # get rid of pre-ictal
                # yield X_train, Y_train

                X = np.concatenate((X, X_train))
                X = X.reshape(X.shape[0],size_in,  23,1)
                Y = np.concatenate((Y, Y_train))
        # shuffle data
        permuted_indexes = np.random.permutation(Y.shape[0])
        X = X[permuted_indexes, :, :, :]
        Y = Y[permuted_indexes]
        return X, Y
    else:
        # take only the one for testing
        #(X, Y) = folder_to_dataset(patient_folder + '/' + 'chb' + str(patient_number).zfill(2) + '_' + str(leaveout_sample).zfill(2),size_in)
        mat_var = loadmat(
        patient_folder + '/' + 'chb' + str(patient_number).zfill(2) + '_' + str(leaveout_sample).zfill(
                2) + '_seg.mat')
        X = mat_var['total_images']
        X = X.reshape(X.shape[0],size_in,  23,1)
        Y = mat_var['total_labels']
        X = np.compress((Y != 2).flatten(), X, 0) # get rid of pre-ictal
        Y = np.compress((Y != 2).flatten(), Y, 0) # get rid of pre-ictal
        #Y[Y==2]=0
        #Y = np_utils.to_categorical(Y, 2)
        return X, Y
        # yield X_test, Y_test


if __name__ == "__main__":
    #main_folder = '/home/gustavo/'
    main_folder = '/home/gchau/Documents/data/epilepsia_data_subset/Data_segmentada_ds/'
    batch_size = 200
    num_classes = 2
    epochs = 50
    size_in = 256

    model = Sequential()
    model.add(Conv2D(kernel_size=(30, 1), filters=40), input_shape=(size_in, num_channels, 1))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 1)))
    model.add(Dropout(0.3))
    model.add(Conv2D(kernel_size=(15, 1), filters=18))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 1)))
    model.add(Dropout(0.3))
    model.add(Conv2D(kernel_size=(7, 1), filters=12))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 1)))
    model.add(Dropout(0.3))
    model.add(Flatten())
    model.add(BatchNormalization())
    model.add(Dense(40))  # ,kernel_regularizer=regularizers.l1(0.01)))
    model.add(Activation('relu'))
    model.add(Dropout(0.3))
    model.add(Dense(30))  # ,kernel_regularizer=regularizers.l1(0.01)))
    model.add(Activation('relu'))
    model.add(Dropout(0.3))
    model.add(Dense(num_classes, activation='softmax'))
    model.summary()

    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop')


    los = 4
    patient_number = 1
    X_train, Y_train = data_generator_one_patient(main_folder=main_folder, patient_number=patient_number, size_in=size_in, leaveout_sample=los,
                                                  isTrain=True)

    print(X_train.shape)
    print(Y_train.shape)

    X_test, Y_test = data_generator_one_patient(main_folder=main_folder, patient_number=patient_number, size_in=size_in, leaveout_sample=los,
                                                isTrain=False)


    num_positive = sum(Y_train==1)
    num_negative = sum(Y_train==0)

    class_weight = {0: 1.0,
                   1: 1.0} #*float(num_negative) / float(num_positive)}

    model.compile(loss='categorical_crossentropy',
                  optimizer='sgd',
                  metrics=[metrics.categorical_accuracy])

    Y_train=np_utils.to_categorical(Y_train,2)
    early_stopping = EarlyStopping(monitor='categorical_accuracy', patience=3)

    history = model.fit(X_train, Y_train,
                        batch_size=batch_size,
                        epochs=epochs, #validation_split=0.1,
                        verbose=1, class_weight=class_weight) #, callbacks=[early_stopping])


    Y_test = np_utils.to_categorical(Y_test,2)
    score = model.evaluate(X_test, Y_test, verbose=1)

    print('=== Training ====')
    y_pred_t = np.argmax(model.predict(X_train, verbose=0), axis=1)
    y_true_t = np.argmax(Y_train, axis=1)
    sensitivity, fp = comp_metric(y_true_t, y_pred_t)
    print('Test sensitivity:', sensitivity)
    print('Test # false positives:', float(fp)/(float(X_test.shape[0]/30.0)))

    print('=== Test ====')

    y_pred = np.argmax(model.predict(X_test, verbose=0),axis=1)
    y_true = np.argmax(Y_test,axis=1)
    sensitivity, fp = comp_metric(y_true, y_pred)
    print('Test sensitivity:', sensitivity)
    print('Test # false positives:', float(fp)/(float(X_test.shape[0]/30.0)))

    #plt.plot(y_pred)
    #plt.plot(y_true)
