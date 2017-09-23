'''Trains a CNN over the time segments of data
'''
from __future__ import print_function

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, BatchNormalization, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.optimizers import SGD
from os import listdir
from scipy.io import loadmat
from keras.utils import np_utils
from keras import backend as K
from keras import regularizers
import matplotlib.pyplot as plt
import numpy as np
import math
from keras import metrics
from keras.callbacks import EarlyStopping
from keras.layers import LSTM
from keras.layers.wrappers import TimeDistributed

def create_class_weight(labels_dict,mu=0.15):
    total = np.sum(labels_dict.values())
    keys = labels_dict.keys()
    class_weight = dict()
    for key in keys:
        score = math.log(mu*total/float(labels_dict[key]))
        class_weight[key] = score if score > 1.0 else 1.0
    return class_weight

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

def folder_to_dataset(folder, size_in):
    list_files = listdir(folder)
    # print(list_files)
    num_samples = len(list_files)
    X = np.zeros(shape=(num_samples, size_in, 23)) #dataset in theano format
    Y = np.zeros(shape=(num_samples, 1))  # dataset in theano format

    for index, file in enumerate(list_files):
        # print(folder +'/'+ file)
        try:
            mat_var = loadmat(folder + '/' + file)
        except:
            print(folder + '/' + file)
        # X[index,] = np.transpose(mat_var['x'])
        X[index,:] = (mat_var['x']).reshape(size_in,  23)
        if ((mat_var['label'] == 0)):
            Y[index] = 0
        else:  # ictal
            Y[index] = 1
    # Y = np_utils.to_categorical(Y, 2)
    print(X.shape)
    return (X, Y)


def data_generator_one_patient(main_folder, patient_number,size_in, isTrain=True):
    nb_classes = 2
    if (isTrain):
        patient_folder = main_folder + 'A' + str(patient_number).zfill(2) +'/train'
    else:
        patient_folder = main_folder + 'A' + str(patient_number).zfill(2) +'/test'
    print(patient_folder)

    list_samples = listdir(patient_folder)

    print(list_samples)
    X = np.zeros(shape=((0,size_in[1],size_in[2],size_in[3],1)))
    Y = np.zeros(shape=(0, 1))
    for sample in list_samples:
        # if is not the one to be tested

        print(sample)
        #(X_train, Y_train) = folder_to_dataset(patient_folder + '/' + sample,size_in)
        # yield X_train, Y_train
        mat_var = loadmat(patient_folder + '/' + sample)
        X_train = (mat_var['samples']).reshape(size_in[0], size_in[1], size_in[2], size_in[3], 1)
        Y_train = (mat_var['sample_labels']).reshape(size_in[0],1)
        X = np.concatenate((X, X_train))
        Y = np.concatenate((Y, Y_train))
    #Y = np_utils.to_categorical(Y, 2)
    return X, Y
        # yield X_test, Y_test


if __name__ == "__main__":
    main_folder = '/home/gustavo/Documents/MATLAB/bci/ESI/'
    np.random.seed(7)
    batch_size = 300
    num_classes = 4
    epochs = 2
    size_in = 256
    # 24   501   100   100

    num_time_samples = 501
    num_rows = 100
    num_cols = 100
    num_trials = 24
    size_samples = (num_trials,num_time_samples, num_rows,num_cols)

    model = Sequential()
    model.add(TimeDistributed(Conv2D(40, (3, 3)), input_shape=(num_time_samples, num_rows, num_cols,1)))
    model.add(Activation('relu'))
    model.add(TimeDistributed(MaxPooling2D()))
    model.add(Dropout(0.2))
    model.add(TimeDistributed(Conv2D(20, (3, 3))))
    model.add(Activation('relu'))
    model.add(TimeDistributed(MaxPooling2D()))
    model.add(Dropout(0.2))
    model.add(TimeDistributed(Flatten()))
    model.add(TimeDistributed(BatchNormalization()))
    model.add(LSTM(40))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    # model.add(LSTM(1))
    # model.add(Activation('relu'))
    # model.add(Dropout(0.2))
    model.add(Dense(num_classes, activation='softmax'))
    model.summary()

    # sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=[metrics.categorical_accuracy])

    X_train, Y_train = data_generator_one_patient(main_folder=main_folder, patient_number=1, size_in=size_samples,isTrain=True)

    print(X_train.shape)
    print(Y_train.shape)

    # X_test, Y_test = data_generator_one_patient(main_folder=main_folder, patient_number=1, size_in=size_in, leaveout_sample=los,
    #                                               isTrain=False)
    #
    # # scores = model.evaluate_generator(data_generator_mnist(False), val_samples=10000)
    # # print("Baseline Error: %.2f%%" % (100-scores[1]*100))
    # model.summary()
    #
    # num_positive = sum(Y_train==1)
    # num_negative = sum(Y_train==0)
    #
    # # class weight to compensate unbalanced training set
    # # labels_dict = {0: float(num_negative),
    # #                 1: float(num_positive)}
    # # class_weight = create_class_weight(labels_dict, mu=0.15)
    # class_weight = {0: 1.0,
    #                1: float(num_negative) / float(num_positive)}
    #
    # model.compile(loss='categorical_crossentropy',
    #               optimizer='sgd',
    #               metrics=[metrics.categorical_accuracy])
    #
    Y_train=np_utils.to_categorical(Y_train,num_classes)
    # early_stopping = EarlyStopping(monitor='categorical_accuracy', patience=3)
    #
    history = model.fit(X_train, Y_train,
                         batch_size=batch_size,
                         epochs=epochs, validation_split=0.1,
                         verbose=1)
    #
    X_test, Y_test = data_generator_one_patient(main_folder=main_folder, patient_number=1, size_in=size_samples,
                                                  isTrain=False)
    Y_test = np_utils.to_categorical(Y_test,num_classes)
    score = model.evaluate(X_test, Y_test, verbose=1)
    print(score)
    #
    # print('=== Training ====')
    # y_pred_t = np.argmax(model.predict(X_train, verbose=0), axis=1)
    # y_true_t = np.argmax(Y_train, axis=1)
    # sensitivity, fp = comp_metric(y_true_t, y_pred_t)
    # print('Test sensitivity:', sensitivity)
    # print('Test # false positives:', fp)
    #
    # print('=== Test ====')
    #
    # y_pred = np.argmax(model.predict(X_test, verbose=0),axis=1)
    # y_true = np.argmax(Y_test,axis=1)
    # sensitivity, fp = comp_metric(y_true, y_pred)
    # print('Test sensitivity:', sensitivity)
    # print('Test # false positives:', fp)
    #
    # plt.plot(y_pred)
    # plt.plot(y_true)