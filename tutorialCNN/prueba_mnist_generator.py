'''Trains a simple deep NN on the MNIST dataset.
Gets to 98.40% test accuracy after 20 epochs
(there is *a lot* of margin for parameter tuning).
2 seconds per epoch on a K520 GPU.
'''

from __future__ import print_function

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop


def data_generator_mnist(isTrain = True, batchSize = 100):
    nb_classes = 10
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    X_train = X_train.reshape(60000, 784)
    X_test = X_test.reshape(10000, 784)
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255
    print(X_train.shape[0], 'train samples')
    print(X_test.shape[0], 'test samples')

    # convert class vectors to binary class matrices
    Y_train = np_utils.to_categorical(y_train, nb_classes)
    Y_test = np_utils.to_categorical(y_test, nb_classes)

    if(isTrain):
        dataset = (X_train,Y_train)
    else :
        dataset = (X_test, Y_test)

    dataset_size = dataset[0].shape[0]
    i = 0
    while(True):
        yield dataset[0][i:i+batchSize], dataset[1][i:i+batchSize]
        i += batchSize
        if (i+batchSize>dataset_size) :
            i = 0;

model = Sequential()
model.add(Dense(512, input_shape=(784,)))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(10))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(),
              metrics=['accuracy'])

history = model.fit_generator(data_generator_mnist(True, batchSize=100), samples_per_epoch=60000 \
                              , nb_epoch=12, callbacks=[], \
                              validation_data=data_generator_mnist(False, batchSize=100), nb_val_samples=10000,\
                              max_q_size=10)


scores = model.evaluate_generator(data_generator_mnist(False), val_samples=10000)
print("Baseline Error: %.2f%%" % (100-scores[1]*100))
