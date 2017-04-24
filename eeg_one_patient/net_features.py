'''Trains a FC-NN on spectral features of one patient, predicts over that patient
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
from keras import regularizers
import matplotlib.pyplot as plt
import numpy as np
from keras import metrics

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

def data_generator_one_patient(main_folder, patient_number, leaveout_sample, isTrain=True):
    nb_classes = 2
    patient_folder = main_folder + 'chb' + str(patient_number).zfill(2) +'/extracted_features'
    print(patient_folder)
    list_samples = listdir(patient_folder)
    print(list_samples)
    if (isTrain):
        # take all series except for the one for testing
        X = np.zeros(shape=(0, 552))
        Y = np.zeros(shape=(0, 1))
        for sample in list_samples:
            # if is not the one to be tested
            if (sample != ('chb' + str(patient_number).zfill(2) + '_' + str(leaveout_sample).zfill(2) + '_feats.mat')):
                #print(sample)
                mat_var = loadmat(patient_folder + '/' + sample)
                X_train = mat_var['X']
                Y_train = mat_var['Y']
                X = np.concatenate((X, X_train))
                Y = np.concatenate((Y, Y_train))
        #X,Y = balance_dataset(X, Y)
        #Y = np_utils.to_categorical(Y, 2)
        return X, Y
    else:
        # take only the one for testing
        mat_var = loadmat(patient_folder + '/' + 'chb' + str(patient_number).zfill(2) + '_' + str(leaveout_sample).zfill(2) + '_feats.mat')
        X = mat_var['X']
        Y = mat_var['Y']
        #Y = np_utils.to_categorical(Y, 2)
        return X, Y
        # yield X_test, Y_test


if __name__ == "__main__":
    main_folder = '/media/gustavo/TOSHIBA EXT/EEG/Data_segmentada/'

    batch_size = 8000
    num_classes = 2
    epochs = 25

    model = Sequential()
    model.add(Dense(512, activation='sigmoid', input_shape=(552,), kernel_regularizer=regularizers.l2(0.01)))
    model.add(Dense(256, activation='sigmoid', kernel_regularizer=regularizers.l2(0.01)))
    model.add(Dense(32, activation='sigmoid', kernel_regularizer=regularizers.l2(0.01) ))
    model.add(Dense(2, activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer=RMSprop())
    los = 15
    X_train, Y_train = data_generator_one_patient(main_folder=main_folder, patient_number=1, leaveout_sample=los,
                                                  isTrain=True)

    print(X_train.shape)
    print(Y_train.shape)

    X_test, Y_test = data_generator_one_patient(main_folder=main_folder, patient_number=1, leaveout_sample=los,
                                                  isTrain=False)

    # scores = model.evaluate_generator(data_generator_mnist(False), val_samples=10000)
    # print("Baseline Error: %.2f%%" % (100-scores[1]*100))
    model.summary()

    num_positive = sum(Y_train==1)
    num_negative = sum(Y_train==0)

    # class weight to compensate unbalanced training set
    class_weight = {0: 1.,
                    1: num_negative/num_positive}

    model.compile(loss='categorical_crossentropy',
                  optimizer=RMSprop(),
                  metrics=[metrics.categorical_accuracy])

    Y_train=np_utils.to_categorical(Y_train,2)
    history = model.fit(X_train, Y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        verbose=1,
                        class_weight = class_weight)
    # for k in range(0,epochs):
    #     print(k)
    #     X_sub, Y = balance_dataset(X_train, Y_train)
    #     Y_sub = np_utils.to_categorical(Y, 2)
    #     #print(X_sub.shape)
    #     #print(Y_sub.shape)
    #     model.train_on_batch(X_sub, Y_sub)

    Y_test = np_utils.to_categorical(Y_test,2)
    score = model.evaluate(X_test, Y_test, verbose=1)

    print('=== Training ====')
    y_pred = np.argmax(model.predict(X_train, verbose=0), axis=1)
    y_true = np.argmax(Y_train, axis=1)
    sensitivity, fp = comp_metric(y_true, y_pred)
    print('Test sensitivity:', sensitivity)
    print('Test # false positives:', fp)

    print('=== Test ====')

    y_pred = np.argmax(model.predict(X_test, verbose=0),axis=1)
    y_true = np.argmax(Y_test,axis=1)
    sensitivity, fp = comp_metric(y_true, y_pred)
    print('Test sensitivity:', sensitivity)
    print('Test # false positives:', fp)