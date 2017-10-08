'''Trains a CNN with cross patient data
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

from keras.utils import np_utils
from keras import backend as K
from keras import regularizers
#import matplotlib.pyplot as plt
import numpy as np
import math
import pickle
from keras import metrics
from keras.callbacks import EarlyStopping

from keras.layers.wrappers import TimeDistributed
from keras.layers import LSTM
#import myEmbedLayer
import random

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
    return [sensitivity, fp, fn, tp, tn]

def data_generator_one_patient(main_folder, patient_number,num_per_series,size_in,balance=False,bal_ratio=4):
    nb_classes = 2
    patient_folder = main_folder + 'chb' + str(patient_number).zfill(2)
    print(patient_folder)
    list_samples = listdir(patient_folder)
    print(list_samples)  # take all series except for the one for testing
    X_pat = np.zeros(shape=(0, num_per_series, size_in,23,1))
    Y_pat = np.zeros(shape=(0, 1))
    for sample in list_samples:
#        print(sample)
        mat_var = loadmat(main_folder + 'chb' + str(patient_number).zfill(2) + '/' + sample)
        X_train = mat_var['total_series']
        Y_train = mat_var['total_labels']

        X_train = np.reshape(X_train, (X_train.shape[0], num_per_series, size_in,23,1))
        #Y_train[Y_train==2]=0 # relabel pre-seizure segments
        X_pat = np.concatenate((X_pat, X_train))
        Y_pat = np.concatenate((Y_pat, Y_train))
        # Y = np_utils.to_categorical(Y, 2)
    print(sum(Y_pat==0))
    print(sum(Y_pat==1))
    print(sum(Y_pat==2))
    #print((Y_pat!=2).flatten())
    #print((Y_pat!=2).flatten().shape)
    X_pat = np.compress((Y_pat!=2).flatten(),X_pat,0)
    Y_pat = np.compress((Y_pat!=2).flatten(),Y_pat,0)
    print(sum(Y_pat==0))
    print(sum(Y_pat==1))
    print(sum(Y_pat==2))
    if balance:
        num_positive = sum(Y_pat==1)
        #np.random.seed(7)
        ind_negative = np.where(Y_pat==0)[0]
        sel_ind_negative = random.sample(ind_negative, num_positive[0]*bal_ratio)
        not_selected = list(set(ind_negative) - set(sel_ind_negative)) # which rows to remove
        X_pat = np.delete(X_pat,not_selected,0)
        Y_pat = np.delete(Y_pat,not_selected,0)
    
    #X_pat = np.delete(X_pat,np.where(Y_pat==2),0)
    #Y_pat = np.delete(Y_pat,np.where(Y_pat==2),0)    
    return X_pat, Y_pat

def data_generator_all_patients(main_folder, num_per_series, size_in, list_all_patients, leaveout):
    list_leave = list()
    list_leave.append(leaveout)
    list_patients_training = list(set(list_all_patients) - set(list_leave))
    print(list_patients_training)
    X = np.zeros(shape=(0, num_per_series, size_in, 23,1))
    Y = np.zeros(shape=(0, 1))
    for i in list_patients_training:
        X_temp, Y_temp = data_generator_one_patient(main_folder=main_folder, num_per_series=num_per_series, patient_number=i, size_in=size_in, balance=True)
        X = np.concatenate((X, X_temp))
        Y = np.concatenate((Y, Y_temp))
    return X,Y

if __name__ == "__main__":
    #main_folder = '/home/gchau/Documents/data/epilepsia_data_subset/Data_segmentada_ds/'
    main_folder = '/home/gchau/Documents/data/epilepsia_data/Data_segmentada_ds30/'
    np.random.seed(7)
    batch_size = 30
    num_classes = 2
    epochs = 30
    size_in = 128
    num_channels =23
    num_per_series = 30
    model = Sequential()
    model.add(TimeDistributed(Conv2D(kernel_size=(30,1),filters=40), input_shape=(num_per_series,size_in,num_channels,1)))
    model.add(Activation('relu'))
    model.add(TimeDistributed(MaxPooling2D(pool_size=(2,1))))
    model.add(Dropout(0.3))
    model.add(TimeDistributed(Conv2D(kernel_size=(15,1),filters=18)))
    model.add(Activation('relu'))
    model.add(TimeDistributed(MaxPooling2D(pool_size=(2,1))))
    model.add(Dropout(0.3))
    model.add(TimeDistributed(Conv2D(kernel_size=(7,1),filters=12)))
    model.add(Activation('relu'))
    model.add(TimeDistributed(MaxPooling2D(pool_size=(2,1))))
    model.add(Dropout(0.3))
    model.add(TimeDistributed(Flatten()))
    model.add(TimeDistributed(BatchNormalization()))
    model.add(TimeDistributed(Dense(40)))#,kernel_regularizer=regularizers.l1(0.01)))
    model.add(Activation('relu'))
    model.add(Dropout(0.3))
    model.add(TimeDistributed(Dense(30)))#,kernel_regularizer=regularizers.l1(0.01)))
    model.add(Activation('relu'))
    model.add(Dropout(0.3))

    model.add(Bidirectional(LSTM(25,return_sequences=False)))
    model.add(Activation('relu'))
    model.add(Dropout(0.3))
    #model.add(LSTM(15,return_sequences=False))
    model.add(Dense(20))
    model.add(Activation('relu'))
    model.add(Dropout(0.3))
    model.add(Dense(num_classes, activation='softmax'))
    model.summary()

#    model.add(LSTM(20, return_sequences=True))
#    model.add(Activation('relu'))
#    model.add(Dropout(0.2))
#    model.add(LSTM(10))
#    model.add(Activation('relu'))
#    model.add(Dropout(0.2))
#    model.add(Dense(num_classes, activation='softmax'))
#    model.summary()

    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy',metrics=[metrics.mae, metrics.categorical_accuracy],
                  optimizer='rmsprop')

    model.summary()
    #early_stopping = EarlyStopping(monitor='categorical_accuracy', patience=3)

    ## Training

    lop = 1
    #X_train, Y_train = data_generator_one_patient(main_folder=main_folder, patient_number=1, size_in=size_in, leaveout_sample=los,
    #                                              isTrain=True)
    list_all_patients = range(1, 17) + range(18, 24)   
    #list_all_patients = range(1,11)
    list_leave = list()
    list_leave.append(lop)
    list_patients_training = list(set(list_all_patients) - set(list_leave)) # list of patients over which to train


    num_positive = 0
    num_negative = 0
    
    X_train,Y_train = data_generator_all_patients(main_folder=main_folder, num_per_series=num_per_series, size_in=size_in, list_all_patients=list_all_patients, leaveout=lop)

    num_positive += sum(Y_train == 1)
    num_negative += sum(Y_train == 0)

    print(num_positive)
    print(num_negative)

    class_weight = {0: 1.0,
                    1: 1.0}#float(num_negative)/float(num_positive)}

            
    Y_train = np_utils.to_categorical(Y_train, 2)
        #model.train_on_batch(X_train, Y_train, class_weight=class_weight)
    early_stopping = EarlyStopping(monitor='categorical_accuracy', patience=3)
    history = model.fit(X_train, Y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        verbose=1, class_weight=class_weight)#, callbacks=[early_stopping])

    ###### Testing

    X_test, Y_test = data_generator_one_patient(main_folder = main_folder, patient_number=lop, num_per_series=num_per_series, size_in=size_in)
    print(X_test.shape)
    print(Y_test.shape)

    Y_test = np_utils.to_categorical(Y_test,2)
    score = model.evaluate(X_test, Y_test, verbose=1)

    print('=== Training ====')
    y_pred_t = np.argmax(model.predict(X_train, verbose=0), axis=1)
    y_true_t = np.argmax(Y_train, axis=1)
    metrics_t = comp_metric(y_true_t, y_pred_t)
    print('Test sensitivity:', metrics_t[0])
    print('Test false positive rate:', float(metrics_t[1])/(float(num_negative+num_positive)/30.0))

    print('=== Test ====')

    y_pred = np.argmax(model.predict(X_test, verbose=0),axis=1)
    y_true = np.argmax(Y_test,axis=1)
    metrics_test = comp_metric(y_true, y_pred)
    print('Test sensitivity:', metrics_test[0])
#    print('Test # false positives:', metrics_test[1])
    print('Test false positive rate:', float(metrics_test[1])/(float(X_test.shape[0]/30.0)))
    variables_save = dict()
    variables_save['y_pred'] = y_pred
    variables_save['y_true'] = y_true
    variables_save['y_pred_t'] = y_pred_t
    variables_save['y_true_t'] = y_true_t
    variables_save['metrics_t'] = metrics_t
    variables_save['metrics_test'] = metrics_test

    with open('objs.pickle', 'w') as f:  # Python 3: open(..., 'wb')
        pickle.dump(variables_save, f)
    savemat(file_name = 'prueba.mat', mdict=variables_save)
    #model.save('lstm_lop' + str(lop))


#    plt.plot(y_pred)
#    plt.plot(y_true)