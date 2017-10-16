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


def list_seizures_patient(patient_number):
    list_seizures = []
    for i in range(23):
        list_seizures.append([])
    list_seizures[0] = [3, 4, 15, 16, 18, 21, 26]
    list_seizures[1] = [16, 19]
    list_seizures[2] = [1, 2, 3, 4, 34, 35, 36]
    list_seizures[3] = [5, 8, 28]
    list_seizures[4] = [6, 13, 16, 17, 22]
    list_seizures[5] = [1, 4, 9, 10, 13, 18, 24]
    list_seizures[6] = [12, 13, 19]
    list_seizures[7] = [2, 5, 11, 13, 21]
    list_seizures[8] = [6, 8, 19]
    list_seizures[9] = [12, 20, 27, 30, 31, 38, 89]
    list_seizures[10] = [82, 92, 99]
    list_seizures[11] = [6, 8, 9, 10, 11, 23, 33, 36, 38, 42]
    list_seizures[12] = [19, 21, 55, 58, 59, 60, 62]
    list_seizures[13] = [3, 4, 6, 11, 17, 18, 27]
    list_seizures[14] = [6, 10, 15, 17, 20, 22, 28, 31, 40, 46, 49, 52, 54, 62]
    list_seizures[15] = [10, 11, 14, 16, 17]
    list_seizures[17] = [29, 30, 31, 32, 35, 36]
    list_seizures[18] = [28, 29, 30]
    list_seizures[19] = [12, 13, 14, 15, 16, 68]
    list_seizures[20] = [19, 20, 21, 22]
    list_seizures[21] = [20, 25, 38]
    list_seizures[22] = [6, 8, 9]
    return list_seizures[patient_number-1]


def data_generator_one_patient(main_folder, patient_number, size_img, leaveout_sample, isTrain=True, balance=False,bal_ratio=4):
    nb_classes = 2
    patient_folder = main_folder + 'chb' + str(patient_number).zfill(2)
    print(patient_folder)
    list_samples = listdir(patient_folder)
    print(list_samples)
    if (isTrain):
        # take all series except for the one for testing
        X = np.zeros(shape=(0, size_img, size_img, 3))
        Y = np.zeros(shape=(0, 1))
        for sample in list_samples:
            # if is not the one to be tested
            if (sample != ('chb' + str(patient_number).zfill(2) + '_' + str(leaveout_sample).zfill(2))):
                #print(sample)
                mat_var = loadmat(main_folder + 'chb' + str(patient_number).zfill(2) + '/' + sample)
                X_train = mat_var['proj_images']
		#print(X_train.shape)
                X_train = X_train.reshape(X_train.shape[0], size_img, size_img, 3)
                Y_train = mat_var['total_labels']
                X_train = np.compress((Y_train != 2).flatten(), X_train, 0)  # get rid of pre-ictal
                Y_train = np.compress((Y_train != 2).flatten(), Y_train, 0)  # get rid of pre-ictal
                # yield X_train, Y_train

                X = np.concatenate((X, X_train))
                Y = np.concatenate((Y, Y_train))

        if balance:
            num_positive = sum(Y == 1)
            ind_negative = np.where(Y == 0)[0]
            sel_ind_negative = random.sample(ind_negative, num_positive[0] * bal_ratio)
            not_selected = list(set(ind_negative) - set(sel_ind_negative))  # which rows to remove
            X = np.delete(X, not_selected, 0)
            Y = np.delete(Y, not_selected, 0)

        # shuffle data
        permuted_indexes = np.random.permutation(Y.shape[0])
        X = X[permuted_indexes, :, :, :]
        Y = Y[permuted_indexes]
        return X, Y
    else:
        # take only the one for testing
        # (X, Y) = folder_to_dataset(patient_folder + '/' + 'chb' + str(patient_number).zfill(2) + '_' + str(leaveout_sample).zfill(2),size_img)
        mat_var = loadmat(patient_folder + '/' + 'chb' + str(patient_number).zfill(2) + '_' + str(leaveout_sample).zfill(
                2) + '_seg.mat')
        X = mat_var['proj_images']
        X = X.reshape(X.shape[0], size_img, size_img, 3)
        Y = mat_var['total_labels']
        X = np.compress((Y != 2).flatten(), X, axis=0) # get rid of pre-ictal
        Y = np.compress((Y != 2).flatten(), Y, axis=0) # get rid of pre-ictal
        #Y[Y==2]=0
        #Y = np_utils.to_categorical(Y, 2)
        return X, Y
        # yield X_test, Y_test


if __name__ == "__main__":
    #main_folder = '/media/gustavo/TOSHIBA EXT/epilepsia_data/proj_images_ds1/'
    #main_folder = '/home/gchau/data/Data_segmentada_ds1/'
    main_folder = '/home/gchau/Documents/data/epilepsia_data/proj_images_ds1/'
    #    main_folder = '/media/gustavo/TOSHIBA EXT/epilepsia_data/Data_segmentada_ds1/'
    batch_size = 128
    num_classes = 2
    epochs = 50
    size_img = 16
    patient_number = 23

    model = Sequential()
    model.add(Conv2D(kernel_size=(3, 3), filters=32, padding = 'valid', input_shape=(size_img, size_img, 3)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(kernel_size=(3, 3), filters=32, padding='valid', input_shape=(size_img, size_img, 3)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.3))
    model.add(Conv2D(kernel_size=(3, 3), filters=64, padding = 'valid'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(kernel_size=(3, 3), filters=64, padding = 'valid'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
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
    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=[metrics.categorical_accuracy])

    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

    model.save_weights('initial.h5') # initial weights for all cv

    list_seizures = list_seizures_patient(patient_number)
    resumen_train = np.zeros(shape=(len(list_seizures), 2))
    resumen_test = np.zeros(shape=(len(list_seizures), 2))
    for idx,los in enumerate(list_seizures):
        if (idx>0):
            break
        name_save_weights = '2d_weights_single_pat' + str(patient_number) + '_sample' + str(los) +'.h5'
        model_checkpoint = ModelCheckpoint(name_save_weights, monitor='val_categorical_accuracy',
                                           save_best_only=True)
        model.load_weights('initial.h5')  # Reinitialize weights
        X_train, Y_train = data_generator_one_patient(main_folder=main_folder, patient_number=patient_number, size_img=size_img, leaveout_sample=los,
                                                      isTrain=True, balance=True,bal_ratio=4)

        print(X_train.shape)
        print(Y_train.shape)

        X_test, Y_test = data_generator_one_patient(main_folder=main_folder, patient_number=patient_number, size_img=size_img, leaveout_sample=los,
                                                    isTrain=False)


        num_positive = sum(Y_train==1)
        num_negative = sum(Y_train==0)

        class_weight = {0: 1.0,
                       1: 1.0} #*float(num_negative) / float(num_positive)}

        Y_train=np_utils.to_categorical(Y_train,2)

        history = model.fit(X_train, Y_train,
                            batch_size=batch_size,
                            epochs=epochs,
                            shuffle=True,
                            validation_split=0.1,
                            callbacks=[model_checkpoint],
                            verbose=2, class_weight=class_weight)  # , callbacks=[early_stopping])

        Y_test = np_utils.to_categorical(Y_test,2)
        score = model.evaluate(X_test, Y_test, verbose=1)

        print('=== Training ====')
        y_pred_t = np.argmax(model.predict(X_train, verbose=0), axis=1)
        y_true_t = np.argmax(Y_train, axis=1)
        sensitivity, fp = comp_metric(y_true_t, y_pred_t)
        print('Train sensitivity:', sensitivity)
        print('Train # false positives:', float(fp)/(float(X_train.shape[0])/3600.0))

        resumen_train[idx, 0] = sensitivity
        resumen_train[idx, 1] = float(fp)/(float(X_train.shape[0])/3600.0)

        print('=== Test ====')

        y_pred = np.argmax(model.predict(X_test, verbose=0),axis=1)
        y_true = np.argmax(Y_test,axis=1)
        sensitivity, fp = comp_metric(y_true, y_pred)
        print('Test sensitivity:', sensitivity)
        print('Test # false positives:', float(fp)/(float(X_test.shape[0])/3600.0))

        resumen_test[idx, 0] = sensitivity
        resumen_test[idx, 1] = float(fp)/(float(X_test.shape[0])/3600.0)

    promedio_train = np.average(resumen_train, axis=0)
    promedio_test = np.average(resumen_test, axis=0)

    print('===total===')
    print('train: ' + str(promedio_train[0]) + '  ' + str(promedio_train[1]))
    print('test: ' + str(promedio_test[0]) + '  ' + str(promedio_test[1]))

    #plt.plot(y_pred)
    #plt.plot(y_true)
