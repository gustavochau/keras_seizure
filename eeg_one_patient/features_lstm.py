'''Trains a FC-NN on spectral features of one patient, predicts over that patient
'''
from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, BatchNormalization, TimeDistributed
from keras.layers import LSTM
from keras.optimizers import SGD
from os import listdir
from scipy import signal
import scipy as sp
from scipy.io import loadmat
from keras.utils import np_utils
from keras import backend as K
from keras import regularizers
#import matplotlib.pyplot as plt
import numpy as np
import math
from keras import metrics
#from keras.callbacks import EarlyStopping
from keras.layers import LSTM
from myEmbedLayer import myEmbedLayer
from keras.layers import RepeatVector

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

def data_generator_one_patient(main_folder, patient_number, leaveout_sample, isTrain=True):
    nb_classes = 2
    patient_folder = main_folder + 'chb' + str(patient_number).zfill(2)
    print(patient_folder)
    list_samples = listdir(patient_folder)
    print(list_samples)
    if (isTrain):
        # take all series except for the one for testing
        X = np.zeros(shape=(0, 460))
        Y = np.zeros(shape=(0, 1))
        for sample in list_samples:
            # if is not the one to be tested
            if (sample != ('chb' + str(patient_number).zfill(2) + '_' + str(leaveout_sample).zfill(2) + '_seg_feats_wav.mat')):
                #print(sample)
                mat_var = loadmat(patient_folder + '/' + sample)
                X_train = mat_var['X']
                Y_train = mat_var['Y']
                X = np.concatenate((X, X_train))
                Y = np.concatenate((Y, Y_train))
        #X,Y = balance_dataset(X, Y)
        #Y = np_utils.to_categorical(Y, 2)
    else:
        # take only the one for testing
        mat_var = loadmat(patient_folder + '/' + 'chb' + str(patient_number).zfill(2) + '_' + str(leaveout_sample).zfill(2) + '_seg_feats_wav.mat')
        X = mat_var['X']
        Y = mat_var['Y']
        #Y = np_utils.to_categorical(Y, 2)

    # stack time samples
    x1 = (X[0:-3,:]).reshape((-1,1,460))
    x2 = (X[1:-2,:]).reshape((-1,1,460))
    x3 = (X[2:-1,:]).reshape((-1,1,460))
    Y = Y[2:-1,:]
    X = np.concatenate((x1,x2,x3),axis=1)
    return X, Y
        # yield X_test, Y_test


if __name__ == "__main__":
    main_folder = '/media/gustavo/TOSHIBA EXT/EEG/epilepsia/Data_segmentada_ds/extracted_features/'
    np.random.seed(7)
    batch_size = 1000
    num_classes = 2
    epochs = 50

    model = Sequential()
    model.add(LSTM(100,input_shape=(3,460), ))
    model.add(BatchNormalization(input_shape=(460,)))
    model.add(Dropout(0.4))
    model.add(Dense(300, activation='relu', input_shape=(460,)))
    model.add(Dropout(0.2))
    model.add(Dense(120, activation='relu',))
    model.add(Dropout(0.2))
    model.add(Dense(50, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(2, activation='softmax'))

    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

    pnum = 13
    los  = 21
    X_train, Y_train = data_generator_one_patient(main_folder=main_folder, patient_number=pnum, leaveout_sample=los,
                                                  isTrain=True)

    print(X_train.shape)
    print(Y_train.shape)

    X_test, Y_test = data_generator_one_patient(main_folder=main_folder, patient_number=pnum, leaveout_sample=los,
                                                  isTrain=False)

    # scores = model.evaluate_generator(data_generator_mnist(False), val_samples=10000)
    # print("Baseline Error: %.2f%%" % (100-scores[1]*100))
    model.summary()

    num_positive = sum(Y_train==1)
    num_negative = sum(Y_train==0)

    # class weight to compensate unbalanced training set
    # labels_dict = {0: float(num_negative),
    #                 1: float(num_positive)}
    # class_weight = create_class_weight(labels_dict, mu=0.15)
    class_weight = {0: 1.0,
                   1: float(num_negative) / float(num_positive)}


    model.compile(loss='categorical_crossentropy',
                  optimizer='sgd',
                  metrics=[metrics.categorical_accuracy])

    Y_train = np_utils.to_categorical(Y_train,2)
    #early_stopping = EarlyStopping(monitor='categorical_accuracy', patience=3)

    history = model.fit(X_train, Y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        verbose=1, class_weight=class_weight)


    Y_test = np_utils.to_categorical(Y_test,2)
    score = model.evaluate(X_test, Y_test, verbose=1)

    print('=== Training ====')
    y_pred_t = np.argmax(model.predict(X_train, verbose=0), axis=1)
    y_true_t = np.argmax(Y_train, axis=1)
    sensitivity, fp = comp_metric(y_true_t, y_pred_t)
    print('Test sensitivity:', sensitivity)
    print('Test # false positives:', fp)

    print('=== Test ====')

    y_pred = np.argmax(model.predict(X_test, verbose=0),axis=1)
    y_true = np.argmax(Y_test,axis=1)
    sensitivity, fp = comp_metric(y_true, y_pred)
    print('Test sensitivity:', sensitivity)
    print('Test # false positives:', fp)

    #plt.plot(y_pred)
    #plt.plot(y_true)
