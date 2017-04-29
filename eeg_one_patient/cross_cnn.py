'''Trains a CNN with cross patient data
'''

from __future__ import print_function

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, BatchNormalization, Flatten
from keras.layers.convolutional import Conv1D, MaxPooling1D
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

def data_generator_all_patients(main_folder, size_in, leaveout, istrain):
    if istrain:
        #list_all_patients = range(1, 17) + range(18, 24)
        list_all_patients = [1,3,12,15]
        list_leave = list()
        list_leave.append(leaveout)
        list_patients_training = list(set(list_all_patients) - set(list_leave))
        print(list_patients_training)
        X = np.zeros(shape=(0, size_in, 23))
        Y = np.zeros(shape=(0, 1))
        for i in list_patients_training:
            X_temp, Y_temp = data_generator_one_patient(main_folder=main_folder, patient_number=i, size_in=size_in)
            X = np.concatenate((X, X_temp))
            Y = np.concatenate((Y, Y_temp))
    else:
        X,Y = data_generator_one_patient(main_folder = main_folder, patient_number=leaveout ,size_in=size_in)
    return X,Y

def data_generator_one_patient(main_folder, patient_number,size_in):
    nb_classes = 2
    patient_folder = main_folder + 'chb' + str(patient_number).zfill(2)
    print(patient_folder)
    list_samples = listdir(patient_folder)
    print(list_samples)  # take all series except for the one for testing
    X_pat = np.zeros(shape=(0, size_in, 23))
    Y_pat = np.zeros(shape=(0, 1))
    for sample in list_samples:
        print(sample)
        mat_var = loadmat(main_folder + 'chb' + str(patient_number).zfill(2) + '/' + sample)
        X_train = mat_var['total_images']
        Y_train = mat_var['total_labels']
        # yield X_train, Y_train
        X_pat = np.concatenate((X_pat, X_train))
        Y_pat = np.concatenate((Y_pat, Y_train))
        # Y = np_utils.to_categorical(Y, 2)
    return X_pat, Y_pat



if __name__ == "__main__":
    main_folder = '/media/gustavo/TOSHIBA EXT/EEG/epilepsia/Data_segmentada_ds/'
    np.random.seed(7)
    batch_size = 1000
    num_classes = 2
    epochs = 50
    size_in = 256

    model = Sequential()
    model.add(Conv1D(nb_filter=30, filter_length=8, input_shape=(size_in,23)))
    model.add(Activation('relu'))
    model.add(MaxPooling1D())
    model.add(Dropout(0.2))
    model.add(Conv1D(nb_filter=20, filter_length=5))
    model.add(Activation('relu'))
    model.add(MaxPooling1D())
    model.add(Dropout(0.2))
    model.add(Conv1D(nb_filter=10, filter_length=3))
    model.add(Activation('relu'))
    model.add(MaxPooling1D())
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(BatchNormalization())
    model.add(Dense(50, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(25, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(10, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(num_classes, activation='softmax'))

    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy',
                  optimizer='sgd')
    lop = 1
    #X_train, Y_train = data_generator_one_patient(main_folder=main_folder, patient_number=1, size_in=size_in, leaveout_sample=los,
    #                                              isTrain=True)

    X_train, Y_train = data_generator_all_patients(main_folder=main_folder , size_in =size_in, leaveout = lop, istrain=True)
    print(X_train.shape)
    print(Y_train.shape)

    X_test, Y_test = data_generator_all_patients(main_folder=main_folder, size_in=size_in, leaveout=lop, istrain=False)
    print(X_test.shape)
    print(Y_test.shape)

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

    Y_train=np_utils.to_categorical(Y_train,2)
    early_stopping = EarlyStopping(monitor='categorical_accuracy', patience=3)

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

    plt.plot(y_pred)
    plt.plot(y_true)