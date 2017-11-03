import numpy as np
from os import listdir
from scipy.io import loadmat,savemat
import random

# obtain metrics of sensitivity and fpr
def comp_metric(y_true, y_pred, time_samples):
    fp = sum(np.logical_and(y_true == 0, y_pred == 1))
    fn = sum(np.logical_and(y_true == 1, y_pred == 0))
    tp = sum(np.logical_and(y_true == 1, y_pred == 1))
    tn = sum(np.logical_and(y_true == 0, y_pred == 0))
    sensitivity = 100.0 * float(tp) / float((tp + fn))
    fpr = float(fp) / (float(len(y_true)*time_samples)/3600.00) #fpr in events per hour
    return [sensitivity, fpr]


def data_generator_one_patient(main_folder, patient_number, num_per_series, size_in, balance=False, bal_ratio=4):
    patient_folder = main_folder + 'chb' + str(patient_number).zfill(2)
    print(patient_folder)
    list_samples = listdir(patient_folder)
    print(list_samples)  # take all series except for the one for testing
    X_pat = np.zeros(shape=(0, num_per_series, size_in, 23, 1))
    Y_pat = np.zeros(shape=(0, 1))
    for sample in list_samples:
        #        print(sample)
        mat_var = loadmat(main_folder + 'chb' + str(patient_number).zfill(2) + '/' + sample)
        X_train = mat_var['total_series']
        Y_train = mat_var['total_labels']

        X_train = np.reshape(X_train, (X_train.shape[0], num_per_series, size_in, 23, 1))
        # Y_train[Y_train==2]=0 # relabel pre-seizure segments
        X_pat = np.concatenate((X_pat, X_train))
        Y_pat = np.concatenate((Y_pat, Y_train))
        # Y = np_utils.to_categorical(Y, 2)
    print(sum(Y_pat == 0))
    print(sum(Y_pat == 1))
    print(sum(Y_pat == 2))
    # print((Y_pat!=2).flatten())
    # print((Y_pat!=2).flatten().shape)
    X_pat = np.compress((Y_pat != 2).flatten(), X_pat, 0)
    Y_pat = np.compress((Y_pat != 2).flatten(), Y_pat, 0)
    print(sum(Y_pat == 0))
    print(sum(Y_pat == 1))
    print(sum(Y_pat == 2))
    if balance:
        num_positive = sum(Y_pat == 1)
        # np.random.seed(7)
        ind_negative = np.where(Y_pat == 0)[0]
        sel_ind_negative = random.sample(ind_negative, num_positive[0] * bal_ratio)
        not_selected = list(set(ind_negative) - set(sel_ind_negative))  # which rows to remove
        X_pat = np.delete(X_pat, not_selected, 0)
        Y_pat = np.delete(Y_pat, not_selected, 0)

    # X_pat = np.delete(X_pat,np.where(Y_pat==2),0)
    # Y_pat = np.delete(Y_pat,np.where(Y_pat==2),0)
    return X_pat, Y_pat


def data_generator_all_patients(main_folder, num_per_series, size_in, list_all_patients, leaveout):
    list_leave = list()
    list_leave.append(leaveout)
    list_patients_training = list(set(list_all_patients) - set(list_leave))
    print(list_patients_training)
    X = np.zeros(shape=(0, num_per_series, size_in, 23, 1))
    Y = np.zeros(shape=(0, 1))
    pat_indicator = np.zeros(shape=(0, 1))
    for i in list_patients_training:
        X_temp, Y_temp = data_generator_one_patient(main_folder=main_folder, num_per_series=num_per_series,
                                                    patient_number=i, size_in=size_in, balance=True)
        X = np.concatenate((X, X_temp))
        Y = np.concatenate((Y, Y_temp))
        pat_indicator = np.concatenate((pat_indicator, i * np.ones(shape=(X_temp.shape[0], 1))))

    # shuffle data
    permuted_indexes = np.random.permutation(Y.shape[0])
    X = X[permuted_indexes, :, :, :]
    Y = Y[permuted_indexes]
    pat_indicator = pat_indicator[permuted_indexes]
    return X, Y, pat_indicator