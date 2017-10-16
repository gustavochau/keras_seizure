'''Load weights of convolutional layers obtained from all patients. Freeze them and train 
the lstm part over them for each specific patient. I want to take look at the weights
 of the lstm part for the patients with low performance
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

def data_generator_one_patient(main_folder, patient_number, num_per_series, size_img, balance=False, bal_ratio=4):
    nb_classes = 2
    patient_folder = main_folder + 'chb' + str(patient_number).zfill(2)
    print(patient_folder)
    list_samples = listdir(patient_folder)
    print(list_samples)  # take all series except for the one for testing
    X_pat = np.zeros(shape=(0, num_per_series, size_img, size_img, 3))
    Y_pat = np.zeros(shape=(0, 1))
    for sample in list_samples:
#        print(sample)
        mat_var = loadmat(main_folder + 'chb' + str(patient_number).zfill(2) + '/' + sample)
        X_train = mat_var['proj_images']
        Y_train = mat_var['total_labels']

        X_train = np.reshape(X_train, (X_train.shape[0], num_per_series, size_img, size_img, 3))
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
    #Y_pat[Y_pat==2]=0
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

def data_generator_all_patients(main_folder, num_per_series, size_img, list_all_patients, leaveout):
    list_leave = list()
    list_leave.append(leaveout)
    list_patients_training = list(set(list_all_patients) - set(list_leave))
    print(list_patients_training)
    X = np.zeros(shape=(0, num_per_series, size_img, size_img, 3))
    Y = np.zeros(shape=(0, 1))
    for i in list_patients_training:
        X_temp, Y_temp = data_generator_one_patient(main_folder=main_folder, num_per_series=num_per_series, patient_number=i, size_img=size_img, balance=True)
        X = np.concatenate((X, X_temp))
        Y = np.concatenate((Y, Y_temp))
    # shuffle data
    permuted_indexes = np.random.permutation(Y.shape[0])    
    X = X[permuted_indexes,:,:,:]
    Y = Y[permuted_indexes]
    return X,Y

if __name__ == "__main__":
    #main_folder = '/home/gchau/Documents/data/epilepsia_data_subset/Data_segmentada_ds/'
    main_folder = '/home/gchau/Documents/data/epilepsia_data/proj_images_ds30/'

    os.environ['PYTHONHASHSEED'] = '0'
    #session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
    #sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
    #K.set_session(sess)
    batch_size = 30
    num_classes = 2
    epochs = 40
    size_img = 16
    num_channels =23
    num_per_series = 30
    lop = 6
    model = Sequential()
    model.add(TimeDistributed(Conv2D(kernel_size=(3,3),filters=32), input_shape=(num_per_series, size_img, size_img, 3),name='conv1',trainable=False))
    #model.add(Activation('relu'))
    model.add(Dropout(0.3))
    model.add(TimeDistributed(Conv2D(kernel_size=(3,3),filters=32),name='conv2',trainable=False))
    #model.add(Activation('relu'))
    model.add(TimeDistributed(MaxPooling2D(pool_size=(2,2))))
    model.add(Dropout(0.3))
    model.add(TimeDistributed(Conv2D(kernel_size=(3,3),filters=64),name='conv3', trainable=False))
    #model.add(Activation('relu'))
    model.add(Dropout(0.3))
    model.add(TimeDistributed(Conv2D(kernel_size=(3,3),filters=64),name='conv4', trainable=False))
    #model.add(Activation('relu'))
    model.add(TimeDistributed(MaxPooling2D(pool_size=(2,2))))
    model.add(Dropout(0.3))
    model.add(TimeDistributed(Flatten()))
    model.add(TimeDistributed(BatchNormalization(trainable=False)))
    model.add(Bidirectional(LSTM(128,return_sequences=False)))
    model.add(Activation('relu'))
    model.add(Dropout(0.3))
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.3))
    model.add(Dense(num_classes, activation='softmax', name = 'output_layer'))
    model.summary()



    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy',metrics=[metrics.mae, metrics.categorical_accuracy],
                  optimizer='rmsprop')

    model.summary()
    name_pretrained_weights = 'cross2d_pat' + str(18) + '_weights.h5'
    model.load_weights(name_pretrained_weights, by_name=True)
    model.save_weights('initial.h5')

    #early_stopping = EarlyStopping(monitor='categorical_accuracy', patience=3)

    ## Training


    #X_train, Y_train = data_generator_one_patient(main_folder=main_folder, patient_number=1, size_img=size_img, leaveout_sample=los,
    #                                              isTrain=True)
    list_all_patients = range(1, 17) + range(18, 24)
    #list_all_patients = range(1,5)
    # list_leave = list()
    # list_leave.append(lop)
    # list_patients_training = list(set(list_all_patients) - set(list_leave)) # list of patients over which to train

    class_weight = {0: 1.0,
                    1: 1.0} #float(num_negative)/float(num_positive)}



    ###### Load training data
    num_patients = len(list_all_patients)
    resumen_train = np.zeros(shape=(num_patients,2))
    resumen_test = np.zeros(shape=(num_patients,2))

    for zz,lop in enumerate(list_all_patients):
        X_train, Y_train = data_generator_one_patient(main_folder=main_folder, patient_number=lop,
                                                      num_per_series=num_per_series, size_img=size_img)
        Y_train = np_utils.to_categorical(Y_train, 2)
        nombre_pesos_save = 'lstm_experiment_pat' + str(lop) + '_weights.h5'
        model_checkpoint = ModelCheckpoint(nombre_pesos_save, monitor='val_categorical_accuracy', save_best_only=True)
        model.load_weights('initial.h5')  # Reinitialize weights
        history = model.fit(X_train, Y_train,
                            batch_size=batch_size,
                            epochs=epochs,
                            shuffle=True,
                            validation_split=0.0,
                            callbacks=[model_checkpoint],
                            verbose=2, class_weight=class_weight)  # , callbacks=[early_stopping])
        #model.load_weights(nombre_pesos_save)
        print('=== Training ====')
        y_pred_t = np.argmax(model.predict(X_train, verbose=0), axis=1)
        y_true_t = np.argmax(Y_train, axis=1)
        metrics_t = comp_metric(y_true_t, y_pred_t)
        print('Train sensitivity:', metrics_t[0])
        print('Train false positive rate:', float(metrics_t[1]) / (float(X_train.shape[0])*30.0 / 3600.0))

        resumen_train[zz, 0] = metrics_t[0]
        resumen_train[zz, 1] = float(metrics_t[1]) / (float(X_train.shape[0])*30.0 / 3600.0)



    #model.save('lstm_lop' + str(lop))


#    plt.plot(y_pred)
#    plt.plot(y_true)