'''Trains a CNN with cross patient data
'''

from __future__ import print_function
from keras import regularizers
from keras.layers import Bidirectional
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, BatchNormalization, Flatten, Permute, Lambda, AveragePooling2D, Reshape
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

def data_generator_one_patient(main_folder, patient_number,num_per_series,size_in,balance=False,bal_ratio=1):
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
    pat_indicator = np.zeros(shape=(0, 1))
    for i in list_patients_training:
        X_temp, Y_temp = data_generator_one_patient(main_folder=main_folder, num_per_series=num_per_series, patient_number=i, size_in=size_in, balance=True)
        X = np.concatenate((X, X_temp))
        Y = np.concatenate((Y, Y_temp))
        pat_indicator = np.concatenate((pat_indicator, i*np.ones(shape=(X_temp.shape[0],1)) ))

    # shuffle data
    permuted_indexes = np.random.permutation(Y.shape[0])    
    X = X[permuted_indexes,:,:,:]
    Y = Y[permuted_indexes]
    pat_indicator = pat_indicator[permuted_indexes]
    return X,Y,pat_indicator

if __name__ == "__main__":
    #main_folder = '/home/gchau/Documents/data/epilepsia_data_subset/Data_segmentada_ds/'
    main_folder = '/home/gchau/Documents/data/epilepsia_data/Data_segmentada_ds30/'
    os.environ['PYTHONHASHSEED'] = '0'
    #session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
    #sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
    #K.set_session(sess)


    batch_size = 20
    num_classes = 2
    epochs = 1
    size_in = 128
    num_channels =23
    num_per_series = 30
    nb_filters = 10


    contenedor = np.load('30seg.npz')
    X_data_all = contenedor['X']
    Y_data_all = contenedor['Y']
    pat_indicator = contenedor['indic']
    print('todo: ' + str(X_data_all.shape))
    num_realizations = 1
    list_all_patients = list(set(range(1, 17) + range(18, 24)) - set([6, 12, 14, 16]))
    num_other_patients = len(list_all_patients)-1
    results_summary = np.zeros(shape=(num_other_patients, 4, num_realizations))

    for lop in [1]: #list_all_patients:
        list_patients_training = list(set(list_all_patients)-set([lop]))  # list patients
        print('=== processing patient' + str(lop) + '=====')

        # Define model
        model = Sequential()
        model.add(
            TimeDistributed(Conv2D(kernel_size=(30, 1), filters=nb_filters),
                            input_shape=(num_per_series, size_in, num_channels, 1),
                            name='conv1'))  # model frequency filters
        model.add(Lambda(lambda x: x ** 2))  # energy
        model.add(TimeDistributed(AveragePooling2D(pool_size=(99, 1))))
        model.add(Permute((1, 4, 3, 2)))
        model.add(Reshape((num_per_series, nb_filters, num_channels)))  # series x bands x channels
        model.add(TimeDistributed(Flatten()))
        model.add(Bidirectional(LSTM(30, return_sequences=False), name='lstm'))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
        model.add(Dense(15, name='fc1-l'))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
        model.add(Dense(num_classes, activation='softmax', name='output_layer'))
        model.compile(loss='categorical_crossentropy',metrics=[metrics.mae, metrics.categorical_accuracy],
                      optimizer='rmsprop')
        model.save_weights('initials.h5')

        class_weight = {0: 1.0,
                        1: 1.0} #float(num_negative)/float(num_positive)}

        for zz in range(0, num_realizations):
            # each realization takes a different order of the patients
            permuted_indexes = np.random.permutation(num_other_patients)

            take_patient_order = [list_patients_training[ii] for ii in permuted_indexes]
            print('all: '+str(take_patient_order))
            for pp in range(0,4): #range(0,num_other_patients):
                print('Taking ' + str(pp+1) + ' patient(s)')
                subset_patients = take_patient_order[0:pp+1] # take first pp+1 patients
                print(subset_patients)
                # separate in training and testing for this patient
                logic_list = [pat in subset_patients for pat in list(pat_indicator)]
                print(sum(logic_list))
                X_train = np.delete(X_data_all, np.where(logic_list), axis=0)
                Y_train = np.delete(Y_data_all, np.where(logic_list), axis=0)
                X_test = np.compress((pat_indicator == lop).flatten(), X_data_all, 0)
                Y_test = np.compress((pat_indicator == lop).flatten(), Y_data_all, 0)
                print('train: ' + str(X_train.shape))
                print('test: ' + str(X_test.shape))

                Y_train = np_utils.to_categorical(Y_train, 2)
                Y_test = np_utils.to_categorical(Y_test, 2)

                resumen_train = np.zeros(shape=(num_realizations,2))
                resumen_test = np.zeros(shape=(num_realizations,2))


                model.load_weights('initials.h5')
                nombre_pesos_save = 'c1_lstm_pat' + str(lop) + '_weights_taking' + str(pp+1) + '.h5'
                model_checkpoint = ModelCheckpoint(nombre_pesos_save, monitor='val_categorical_accuracy', save_best_only=True)
                history = model.fit(X_train, Y_train,
                                    batch_size=batch_size,
                                    epochs=epochs,
                                    shuffle=True,
                                    validation_split=0.2,
                                    callbacks=[model_checkpoint],
                                    verbose=1, class_weight=class_weight)  # , callbacks=[early_stopping])
                model.load_weights(nombre_pesos_save)
                score = model.evaluate(X_test, Y_test, verbose=1)

                print('=== Training ====')
                y_pred_t = np.argmax(model.predict(X_train, verbose=0), axis=1)
                y_true_t = np.argmax(Y_train, axis=1)
                metrics_t = comp_metric(y_true_t, y_pred_t)
                print('Train sensitivity:', metrics_t[0])
                print('Train false positive rate:', float(metrics_t[1]) / (float(X_train.shape[0]) * 30.0 / 3600.0))

                resumen_train[zz, 0] = metrics_t[0]
                resumen_train[zz, 1] = float(metrics_t[1]) / (float(X_train.shape[0]) * 30.0 / 3600.0)

                results_summary[pp, 0, zz] = metrics_t[0]
                results_summary[pp, 1, zz] = float(metrics_t[1]) / (float(X_train.shape[0]) * 30.0 / 3600.0)

                print('=== Test ====')
                y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
                y_true = np.argmax(Y_test, axis=1)
                metrics_test = comp_metric(y_true, y_pred)
                print('Test sensitivity:', metrics_test[0])
                #    print('Test # false positives:', metrics_test[1])
                print('Test false positive rate:', float(metrics_test[1]) / (float(X_test.shape[0]) * 30.0 / 3600.0))

                resumen_test[zz, 0] = metrics_test[0]
                resumen_test[zz, 1] = float(metrics_test[1]) / (float(X_test.shape[0]) * 30.0 / 3600.0)
                results_summary[pp, 2, zz] = metrics_test[0]
                results_summary[pp, 3, zz] = float(metrics_test[1]) / (float(X_test.shape[0]) * 30.0 / 3600.0)

                promedio_train = np.average(resumen_train, axis=0)
                promedio_test = np.average(resumen_test, axis=0)

                print('===total===')
                print('train: ' + str(promedio_train[0]) + '  ' + str(promedio_train[1]))
                print('test: ' + str(promedio_test[0]) + '  ' + str(promedio_test[1]))

    variables_save = dict()
    variables_save['results_summary'] = results_summary
    nombre_save_matlab = 'sequential_patient_lop' + str(lop) +'.mat'
    savemat(file_name=nombre_save_matlab, mdict=variables_save)
