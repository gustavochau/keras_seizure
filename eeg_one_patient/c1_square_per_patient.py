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
#np.random.seed(7)
import math
import pickle
from keras import metrics
from keras.callbacks import EarlyStopping

from keras.layers.wrappers import TimeDistributed
from keras.layers import LSTM
#import myEmbedLayer
import random
#random.seed(45)
import os
import tensorflow as tf
#tf.set_random_seed(45)

from misc_functions import comp_metric

def create_class_weight(labels_dict,mu=0.15):
    total = np.sum(labels_dict.values())
    keys = labels_dict.keys()
    class_weight = dict()
    for key in keys:
        score = math.log(mu*total/float(labels_dict[key]))
        class_weight[key] = score if score > 1.0 else 1.0
    return class_weight

# def comp_metric(y_true, y_pred):
#     fp = sum(np.logical_and(y_true == 0, y_pred == 1))
#     fn = sum(np.logical_and(y_true == 1, y_pred == 0))
#     tp = sum(np.logical_and(y_true == 1, y_pred == 1))
#     tn = sum(np.logical_and(y_true == 0, y_pred == 0))
#     sensitivity = 100.0*float(tp) /float((tp + fn))
#     return [sensitivity, fp, fn, tp, tn]

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

def precision(y_true, y_pred):
    # Calculates the precision
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def recall(y_true, y_pred):
    # Calculates the recall
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def fmeasure(y_true, y_pred):
    # Calculates the f-measure, the harmonic mean of precision and recall.
    return 2. / (1. / recall(y_true, y_pred) + 1. / precision(y_true, y_pred))




def data_generator_one_patient(main_folder, patient_number,num_per_series,size_in, leaveout_sample, isTrain=True,balance = False, bal_ratio = 4):
    nb_classes = 2
    patient_folder = main_folder + 'chb' + str(patient_number).zfill(2)
    print(patient_folder)
    list_samples = listdir(patient_folder)
    print(list_samples)
    if (isTrain):
        # take all series except for the one for testing
        X = np.zeros(shape=(0, num_per_series, size_in, 23,1))
        Y = np.zeros(shape=(0, 1))
        for sample in list_samples:
            # if is not the one to be tested
            if (sample != ('chb' + str(patient_number).zfill(2) + '_' + str(leaveout_sample).zfill(2))):
                #print(sample)
                mat_var = loadmat(main_folder + 'chb' + str(patient_number).zfill(2) + '/' + sample)
                X_train = mat_var['total_series']
		#print(X_train.shape)
                X_train = X_train.reshape(X_train.shape[0],num_per_series,size_in,  23,1)
                Y_train = mat_var['total_labels']
                X_train = np.compress((Y_train != 2).flatten(), X_train, 0)  # get rid of pre-ictal
                Y_train = np.compress((Y_train != 2).flatten(), Y_train, 0)  # get rid of pre-ictal
                # yield X_train, Y_train

                

                X = np.concatenate((X, X_train))
                X = X.reshape(X.shape[0],num_per_series,size_in,  23,1)
                Y = np.concatenate((Y, Y_train))
        
        if balance:
            num_positive = sum(Y == 1)
            # np.random.seed(7)
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
        # (X, Y) = folder_to_dataset(patient_folder + '/' + 'chb' + str(patient_number).zfill(2) + '_' + str(leaveout_sample).zfill(2),size_in)
        mat_var = loadmat(patient_folder + '/' + 'chb' + str(patient_number).zfill(2) + '_' + str(leaveout_sample).zfill(
                2) + '_seg.mat')
        X = mat_var['total_series']
        X = X.reshape(X.shape[0],num_per_series,size_in,  23,1)
        Y = mat_var['total_labels']
        X = np.compress((Y != 2).flatten(), X, 0) # get rid of pre-ictal
        Y = np.compress((Y != 2).flatten(), Y, 0) # get rid of pre-ictal
        #Y[Y==2]=0
        #Y = np_utils.to_categorical(Y, 2)
        return X, Y
        # yield X_test, Y_test



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

if __name__ == "__main__":
    #main_folder = '/home/gchau/Documents/data/epilepsia_data_subset/Data_segmentada_ds/'
    main_folder = '/home/gchau/Documents/data/epilepsia_data/Data_segmentada_ds30/'
    os.environ['PYTHONHASHSEED'] = '0'
    #session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
    #sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
    #K.set_session(sess)
    batch_size = 30
    num_classes = 2
    epochs = 25
    size_in = 128
    num_channels =23
    num_per_series = 30
    nb_filters = 40
    time_sample = 30.0

    list_all_patients = range(1, 17) + range(18, 24)
    #X_data_all,Y_data_all,pat_indicator = data_generator_all_patients(main_folder=main_folder, num_per_series=num_per_series, size_in=size_in, list_all_patients=list_all_patients, leaveout=50)
    #contenedor = np.load('30seg.npz')
    #X_data_all = contenedor['X']
    #Y_data_all = contenedor['Y']
    #pat_indicator = contenedor['indic']
    #print('todo: ' + str(X_data_all.shape))
    #num_realizations = 5

    # Define model
    model = Sequential()
    model.add(
        TimeDistributed(Conv2D(kernel_size=(30, 1), filters=nb_filters),
                        input_shape=(num_per_series, size_in, num_channels, 1),
                        name='conv1'))  # model frequency filters
    model.add(Lambda(lambda x: x ** 2))  # energy
    model.add(TimeDistributed(AveragePooling2D(pool_size=(99, 1)), name='pooling'))
    model.add(Permute((1, 4, 3, 2)))
    model.add(Reshape((num_per_series, nb_filters, num_channels)))  # series x bands x channels
    model.add(Dropout(0.5))
    model.add(TimeDistributed(TimeDistributed(Dense(40, kernel_regularizer=regularizers.l1(0.1))), name='sparse'))
    model.add(TimeDistributed(Flatten(), name='flatten'))
    model.add(BatchNormalization(name = 'no'))
    model.add(Dropout(0.5))
    model.add(Bidirectional(LSTM(128, return_sequences=False, name='lstm')))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    # model.add(Dense(512, name='fc1-l'))
    # model.add(Activation('relu'))
    # model.add(Dropout(0.5))
    model.add(Dense(512, name='fc1'))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax', name='salida'))
    model.compile(loss='categorical_crossentropy', metrics=[metrics.mae, metrics.categorical_accuracy, fmeasure],
                  optimizer='rmsprop')

    results_summary = np.zeros(shape=(24, 4, 40))

    class_weight = {0: 1.0,
                    1: 1.0}  # float(num_negative)/float(num_positive)}

    for patient_number in list_all_patients:
        print('=== processing patient' + str(patient_number) +'=====')

        list_seizures = list_seizures_patient(patient_number)
        resumen_train = np.zeros(shape=(len(list_seizures), 2))
        resumen_test = np.zeros(shape=(len(list_seizures), 2))

        for idx, los in enumerate(list_seizures):

            #if (idx>0): # just to check running of algorithm take first sample for each patient
            #    break

            # load weights trained on other patients
            #name_pretrained_weights = 'c1_energy_lstm_pretrained_pat' + str(patient_number) + '_weights.h5'
            name_pretrained_weights = 'c1_square_onlyconv_' + str(patient_number) + '_weights.h5'

            model.load_weights(name_pretrained_weights, by_name=True)

            # checkpoint, pick best
            name_save_weights = '1d_energy_single_pat' + str(patient_number) + '_sample' + str(los) + '.h5'
            #model_checkpoint = ModelCheckpoint(name_save_weights, monitor='val_categorical_accuracy',
            #                                   save_best_only=True)
            model_checkpoint = ModelCheckpoint(name_save_weights, monitor='val_loss', save_best_only=True)
            X_train, Y_train = data_generator_one_patient(main_folder=main_folder, patient_number=patient_number, size_in=size_in, leaveout_sample=los, num_per_series = num_per_series,
                                                          isTrain=True, balance=True,bal_ratio=4)

            print(X_train.shape)
            print(Y_train.shape)

            X_test, Y_test = data_generator_one_patient(main_folder=main_folder, patient_number=patient_number, size_in = size_in, leaveout_sample=los, num_per_series = num_per_series,
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
                                verbose=0, class_weight=class_weight)  # , callbacks=[early_stopping])

            # load best checkpoint and evaluate
            model.load_weights(name_save_weights)
            Y_test = np_utils.to_categorical(Y_test,2)
            #score = model.evaluate(X_test, Y_test, verbose=1)

            print('=== Training ====')
            y_pred_t = np.argmax(model.predict(X_train, verbose=0), axis=1)
            y_true_t = np.argmax(Y_train, axis=1)
            sensitivity, fpr = comp_metric(y_true_t, y_pred_t, time_sample)
            print('Train sensitivity:', sensitivity)
            print('Train # false positives:', fpr)

            resumen_train[idx, 0] = sensitivity
            resumen_train[idx, 1] = fpr

            results_summary[patient_number,0,idx] = sensitivity
            results_summary[patient_number, 1, idx] = fpr

            print('=== Test ====')

            y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
            y_true = np.argmax(Y_test, axis=1)
            sensitivity, fpr = comp_metric(y_true, y_pred, time_sample)
            print('Test sensitivity:', sensitivity)
            print('Test # false positives:', fpr)

            resumen_test[idx, 0] = sensitivity
            resumen_test[idx, 1] = fpr

            results_summary[patient_number,2,idx] = sensitivity
            results_summary[patient_number,3, idx] = fpr


        promedio_train = np.average(resumen_train, axis=0)
        promedio_test = np.average(resumen_test, axis=0)

        print('===total===')
        print('train: ' + str(promedio_train[0]) + '  ' + str(promedio_train[1]))
        print('test: ' + str(promedio_test[0]) + '  ' + str(promedio_test[1]))

    variables_save = dict()
    variables_save['results_summary'] = results_summary
    savemat(file_name='1d_energy_per_patient.mat', mdict=variables_save)
