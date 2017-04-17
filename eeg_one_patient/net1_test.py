'''Trains a FC-NN on the data of one patient, predicts over that patient
'''
from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import RMSprop
from os import listdir
from scipy.io import loadmat
from keras.utils import np_utils
from keras import backend as K
import matplotlib.pyplot as plt

import numpy as np

def comp_metric(y_true, y_pred):
    fp = sum(np.logical_and(y_true == 0, y_pred == 1))
    fn = sum(np.logical_and(y_true == 1, y_pred == 0))
    tp = sum(np.logical_and(y_true == 1, y_pred == 1))
    tn = sum(np.logical_and(y_true == 0, y_pred == 0))
    sensitivity = tp/(tp + fn)
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

def folder_to_dataset(folder):
    list_files = listdir(folder)
    # print(list_files)
    num_samples = len(list_files)

    X = np.zeros(shape=(num_samples, 512 * 23))
    # X = np.zeros(shape=(num_samples,1,1024,23)) #dataset in theano format
    Y = np.zeros(shape=(num_samples, 1))  # dataset in theano format

    for index, file in enumerate(list_files):
        # print(folder +'/'+ file)
        try:
            mat_var = loadmat(folder + '/' + file)
        except:
            print(folder + '/' + file)
        # X[index,] = np.transpose(mat_var['x'])
        X[index,:] = (mat_var['x']).reshape(1, 512 * 23)
        if ((mat_var['label'] == 0) or (mat_var['label'] == 1)):  # inter or pre ictal
            Y[index] = 0
        else:  # ictal
            Y[index] = 1
    # Y = np_utils.to_categorical(Y, 2)
    print(X.shape)
    return (X, Y)


def data_generator_one_patient(main_folder, patient_number, leaveout_sample, isTrain=True):
    nb_classes = 2
    patient_folder = main_folder + 'chb' + str(patient_number).zfill(2)
    print(patient_folder)
    list_samples = listdir(patient_folder)
    print(list_samples)
    if (isTrain):
        # take all series except for the one for testing
        X = np.zeros(shape=(0, 512 * 23))
        Y = np.zeros(shape=(0, 1))
        for sample in list_samples:
            # if is not the one to be tested
            if (sample != ('chb' + str(patient_number).zfill(2) + '_' + str(leaveout_sample).zfill(2))):
                print(sample)
                (X_train, Y_train) = folder_to_dataset(patient_folder + '/' + sample)
                # yield X_train, Y_train
                X = np.concatenate((X, X_train))
                Y = np.concatenate((Y, Y_train))
        X,Y = balance_dataset(X, Y)
        Y = np_utils.to_categorical(Y, 2)
        return X, Y
    else:
        # take only the one for testing
        (X, Y) = folder_to_dataset(patient_folder + '/' + 'chb' + str(patient_number).zfill(2) + '_' + str(leaveout_sample).zfill(2))
        Y = np_utils.to_categorical(Y, 2)
        return X, Y
        # yield X_test, Y_test


if __name__ == "__main__":
    main_folder = '/media/gustavo/TOSHIBA EXT/EEG/Data_segmentada/'

    batch_size = 100
    num_classes = 2
    epochs = 25

    model = Sequential()
    model.add(Dense(10, input_shape=(23 * 512,)))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(10))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(2))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer=RMSprop(),
                  metrics=['accuracy'])
    los = 16
    X_train, Y_train = data_generator_one_patient(main_folder=main_folder, patient_number=1, leaveout_sample=los,
                                                  isTrain=True)
    # history = model.fit_generator(data_generator_one_patient(main_folder=main_folder, patient_number=1, leaveout_sample=1, isTrain=True, batchSize=100), samples_per_epoch=60000 \
    #                              , nb_epoch=12, callbacks=[])
    print(X_train.shape)
    X_test, Y_test = data_generator_one_patient(main_folder=main_folder, patient_number=1, leaveout_sample=los,
                                                  isTrain=False)

    # scores = model.evaluate_generator(data_generator_mnist(False), val_samples=10000)
    # print("Baseline Error: %.2f%%" % (100-scores[1]*100))
    model.summary()

    model.compile(loss='categorical_crossentropy',
                  optimizer=RMSprop(),
                  metrics=['accuracy'])

    history = model.fit(X_train, Y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        verbose=1)

    score = model.evaluate(X_test, Y_test, verbose=1)

    y_pred = np.argmax(model.predict(X_test, verbose=0),axis=1)
    y_true = np.argmax(Y_test,axis=1)
    sensitivity, fp = comp_metric(y_true, y_pred)

    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    print('Test sensitivity:', sensitivity)
    print('Test # false positives:', fp)