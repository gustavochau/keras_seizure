'''Trains a CNN with cross patient data
'''

from __future__ import print_function
from keras import regularizers
from sklearn.preprocessing import scale
from keras.layers import Bidirectional
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, BatchNormalization, Flatten, Permute
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

def data_generator_one_patient(main_folder, patient_number, size_img, balance=False, bal_ratio=4):
    nb_classes = 2
    patient_folder = main_folder + 'chb' + str(patient_number).zfill(2)
    print(patient_folder)
    list_samples = listdir(patient_folder)
    #print(list_samples)  # take all series except for the one for testing
    X_pat = np.zeros(shape=(0, size_img, size_img, 3))
    Y_pat = np.zeros(shape=(0, 1))
    for sample in list_samples:
#        print(sample)
        mat_var = loadmat(main_folder + 'chb' + str(patient_number).zfill(2) + '/' + sample)
        X_train = mat_var['proj_images']
        Y_train = mat_var['total_labels']

        X_train = np.reshape(X_train, (X_train.shape[0], size_img, size_img, 3))
        #Y_train[Y_train==2]=0 # relabel pre-seizure segments
        X_pat = np.concatenate((X_pat, X_train))
        Y_pat = np.concatenate((Y_pat, Y_train))
        # Y = np_utils.to_categorical(Y, 2)
    #print(sum(Y_pat==0))
    #print(sum(Y_pat==1))
    #print(sum(Y_pat==2))
    #print((Y_pat!=2).flatten())
    #print((Y_pat!=2).flatten().shape)
    X_pat = np.compress((Y_pat!=2).flatten(),X_pat,0)
    Y_pat = np.compress((Y_pat!=2).flatten(),Y_pat,0)
    #print(sum(Y_pat==0))
    #print(sum(Y_pat==1))
    #print(sum(Y_pat==2))
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

def data_generator_all_patients(main_folder, size_img, list_all_patients, leaveout):
    list_leave = list()
    list_leave.append(leaveout)
    list_patients_training = list(set(list_all_patients) - set(list_leave))
    print(list_patients_training)
    X = np.zeros(shape=(0, size_img, size_img, 3))
    Y = np.zeros(shape=(0, 1))
    pat_indicator = np.zeros(shape=(0, 1))
    for i in list_patients_training:
        X_temp, Y_temp = data_generator_one_patient(main_folder=main_folder, patient_number=i, size_img=size_img, balance=True)
        X = np.concatenate((X, X_temp))
        Y = np.concatenate((Y, Y_temp))
        pat_indicator = np.concatenate((pat_indicator, i*np.ones(shape=(X_temp.shape[0],1)) ))

        # suffle data
    permuted_indexes = np.random.permutation(Y.shape[0])    
    X = X[permuted_indexes,:,:,:]
    Y = Y[permuted_indexes]
    pat_indicator = pat_indicator[permuted_indexes]
    return X,Y,pat_indicator



if __name__ == "__main__":
#    main_folder = '/home/gchau/Documents/data/epilepsia_data/Data_segmentada_ds1/'
    #main_folder = '/home/gchau/Documents/data/epilepsia_data/proj_images_ds1/'
    main_folder = '/home/gchau/Documents/data/epilepsia_data/proj_images_polar_ds1/'


    #main_folder = '/home/gchau/data/Data_segmentada_ds1/'
    os.environ['PYTHONHASHSEED'] = '0'
    #session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
    #sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
    #K.set_session(sess)
    batch_size = 128
    num_classes = 2
    epochs = 30
    size_img = 16
    num_channels =23

    #list_all_patients = range(1, 17) + range(18, 24)   
    list_all_patients = [1,2,3,8,14,22]

    # load the patient of all data
    X_data_all,Y_data_all,pat_indicator = data_generator_all_patients(main_folder=main_folder, size_img=size_img, list_all_patients=list_all_patients, leaveout=50)
    #contenedor = np.load('2d_30s.npz')
    #X_data_all = contenedor['X']
    #Y_data_all = contenedor['Y']
    #pat_indicator = contenedor['indic']
    print('todo: ' + str(X_data_all.shape))

    results_summary = np.zeros(shape=(24,4))
    for lop in list_all_patients:
        print('=== processing patient' + str(lop) +'=====')
        # separate in training and testing for this patient
        X_train = np.delete(X_data_all, np.where((pat_indicator==lop).flatten()),axis=0)
        Y_train = np.delete(Y_data_all, np.where((pat_indicator==lop).flatten()),axis=0)
        X_test = np.compress((pat_indicator==lop).flatten(),X_data_all,0)
        Y_test = np.compress((pat_indicator==lop).flatten(),Y_data_all,0)
        print('train: ' +str(X_train.shape))
        print('test: ' +str(X_test.shape))

        model = Sequential()
        model.add(Conv2D(kernel_size=(3, 3), filters=32, padding='valid', input_shape=(size_img, size_img, 3),name='conv1'))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
        model.add(Conv2D(kernel_size=(3, 3), filters=32, padding='valid', input_shape=(size_img, size_img, 3),name='conv2'))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(kernel_size=(3, 3), filters=64, padding='valid',name='conv3'))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
        model.add(Conv2D(kernel_size=(3, 3), filters=64, padding='valid',name='conv4'))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Flatten())
        #model.add(BatchNormalization())
        #model.add(Dense(512, name='fc_cnnpura1'))  # ,kernel_regularizer=regularizers.l1(0.01)))
        #model.add(Activation('relu'))
        #model.add(Dropout(0.5))
        #model.add(Dropout(0.3))
        model.add(Dense(num_classes, activation='softmax', name='fc_cnnpura3'))
        #model.summary()

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

        model.save_weights('initial.h5')

        #early_stopping = EarlyStopping(monitor='categorical_accuracy', patience=3)

        ## Training

        #X_train, Y_train = data_generator_one_patient(main_folder=main_folder, patient_number=1, size_in=size_in, leaveout_sample=los,
        #                                              isTrain=True)
        #list_all_patients = range(1,4)

        num_positive = 0
        num_negative = 0
        

        medias = list()
        desv_est = list()
        for cc in range(0,3):
            medias.append(np.mean(X_train[:, :, :, cc]))
            desv_est.append(np.std(X_train[:, :, :, cc]))
            X_train[:, :,:, cc] = np.reshape(scale(X_train[:, :,:, cc].flatten()), (X_train.shape[0],16, 16))
            #print(str(np.mean(X_train[:,:,:,cc])))
            #print(str(np.std(X_train[:, :, :,cc])))

        num_positive += sum(Y_train == 1)
        num_negative += sum(Y_train == 0)

        #print(num_positive)
        #print(num_negative)

        class_weight = {0: 1.0,
                        1: 4.0} #float(num_negative)/float(num_positive)}

                
        Y_train = np_utils.to_categorical(Y_train, 2)
            #model.train_on_batch(X_train, Y_train, class_weight=class_weight)
        early_stopping = EarlyStopping(monitor='categorical_accuracy', patience=1)

        ###### Load testing data
        #X_test, Y_test = data_generator_one_patient(main_folder = main_folder, patient_number=lop, size_img=size_img)

        for cc in range(0,3):   
            X_test[:,:,:,cc] = X_test[:,:,:,cc]-medias[cc]
            X_test[:, :, :, cc] = (1.0/desv_est[cc])*X_test[:,:,:,cc]
        #print(X_test.shape)
        #print(Y_test.shape)
        Y_test = np_utils.to_categorical(Y_test,2)

        num_realizations = 1
        resumen_train = np.zeros(shape=(num_realizations,2))
        resumen_test = np.zeros(shape=(num_realizations,2))

        for zz in range(0,num_realizations):
            nombre_pesos = 'cross2d_norm_polar_pat' + str(lop) + '_weights_dropall.h5'
            model_checkpoint = ModelCheckpoint(nombre_pesos, monitor='val_categorical_accuracy', save_best_only=True)
            model.load_weights('initial.h5') # Reinitialize weights
            history = model.fit(X_train, Y_train,
                            batch_size=batch_size,
                            epochs=epochs,
                            shuffle=True,
                            validation_split=0.2,
                            callbacks=[model_checkpoint],
                            verbose=0, class_weight=class_weight)#, callbacks=[early_stopping])
            model.load_weights(nombre_pesos)
            model.save_weights(nombre_pesos)
            score = model.evaluate(X_test, Y_test, verbose=1)

            print('=== Training ====')
            y_pred_t = np.argmax(model.predict(X_train, verbose=0), axis=1)
            y_true_t = np.argmax(Y_train, axis=1)
            metrics_t = comp_metric(y_true_t, y_pred_t)
            print('Train sensitivity:', metrics_t[0])
            print('Train false positive rate:', float(metrics_t[1]) / (float(X_train.shape[0])*30.0 / 3600.0))

            resumen_train[zz, 0] = metrics_t[0]
            resumen_train[zz, 1] = float(metrics_t[1]) / (float(X_train.shape[0])*30.0 / 3600.0)

            print('=== Test ====')
            y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
            y_true = np.argmax(Y_test, axis=1)
            metrics_test = comp_metric(y_true, y_pred)
            print('Test sensitivity:', metrics_test[0])
            #    print('Test # false positives:', metrics_test[1])
            print('Test false positive rate:', float(metrics_test[1]) / (float(X_test.shape[0])*30.0 / 3600.0))

            resumen_test[zz, 0] = metrics_test[0]
            resumen_test[zz, 1] = float(metrics_test[1]) / (float(X_test.shape[0])*30.0 / 3600.0)
        
        promedio_train = np.average(resumen_train,axis=0)    
        promedio_test = np.average(resumen_test,axis=0)    

        print('===total===')
        print('train: ' + str(promedio_train[0]) + '  '+ str(promedio_train[1]))
        print('test: ' + str(promedio_test[0]) + '  '+ str(promedio_test[1]))

        results_summary[lop,0] = promedio_train[0]
        results_summary[lop,1] = promedio_train[1]
        results_summary[lop,2] = promedio_test[0]
        results_summary[lop,3] = promedio_test[1]

    variables_save = dict()
    variables_save['results_summary'] = results_summary

        #with open('objs.pickle', 'w') as f:  # Python 3: open(..., 'wb')
        #    pickle.dump(variables_save, f)
    savemat(file_name = 'cnn_only_todo_drop.mat', mdict=variables_save)
        #model.save('lstm_lop' + str(lop))
