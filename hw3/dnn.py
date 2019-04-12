import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers import Conv2D, MaxPooling2D, Flatten, AveragePooling2D
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint

myseed = 10
drop_rate = 0.25
pic_size = 48

'''
#GPU Limition
import os
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF
def get_session(gpu_fraction=0.8):
        
    num_threads = os.environ.get('OMP_NUM_THREADS')
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)
    
    if num_threads:
        return tf.Session(config=tf.ConfigProto(
            gpu_options=gpu_options, intra_op_parallelism_threads=num_threads))
    else:
        return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
'''

'''Addresses'''
X_train_address = sys.argv[1]
save_model_name = sys.argv[2]

'''Function Define'''
def from_dataset():
    #load in by genfromtxt
    dataset = np.genfromtxt(fname=X_train_address, dtype='str', delimiter=' ', skip_header=1)
    dataset = dataset.T
    #fix data into Y_train & X_train
    fixdata = dataset[0]
    Y_train = np.zeros(len(fixdata))
    fix = np.zeros(len(fixdata))
    for n in range(len(fixdata)):
        Y_train[n], fix[n] = fixdata[n].split(',')[0], fixdata[n].split(',')[1]
    dataset[0] = fix
    
    X_train = np.reshape(dataset.T,(28709,pic_size,pic_size,1)).astype('float64')
    X_train /= 255
    np.save('Y_train_label.npy',Y_train)

    #One-hot encoding
    Y_train = np_utils.to_categorical(Y_train, 7)

    '''
    #data normalization
    mean = np.mean(X_train, axis=0)
    std = np.std(X_train, axis=0)
    np.save('mean_std.npy', [mean, std])
    X_train = np.divide(np.subtract(X_train,mean),(std+ 1e-10))
    '''

    #save X_train and Y_train as npy
    np.save('X_train.npy',X_train)
    np.save('Y_train.npy',Y_train)

    return X_train, Y_train

def from_npy():
    return np.load('X_train.npy'), np.load('Y_train.npy')

def read_dataset(n):
    if n == 1:
        return from_dataset()
    else:
        return from_npy()

def show_train_history(train_history, train, validation):
    plt.plot(train_history.history[train])
    plt.plot(train_history.history[validation])
    plt.title('Train History')
    plt.ylabel('train')
    plt.xlabel('Epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()

'''MAIN'''

#limit gpu
#KTF.set_session(get_session())

#read data
X_train, Y_train = read_dataset(10)

#cut valid set
X_train_valid, Y_train_valid = X_train[:3000], Y_train[:3000]
X_train, Y_train = X_train[3000:], Y_train[3000:]

#build model
if 1+1 == 2:

    model = Sequential()
    
    model.add(Flatten(input_shape=(pic_size,pic_size,1)))

    model.add(Dense(1024, kernel_initializer='glorot_normal'))
    model.add(LeakyReLU())

    model.add(Dense(1024, kernel_initializer='glorot_normal'))
    model.add(LeakyReLU())

    model.add(Dense(1024, kernel_initializer='glorot_normal'))
    model.add(LeakyReLU())

    model.add(Dense(1024, kernel_initializer='glorot_normal'))
    model.add(LeakyReLU())

    model.add(Dense(1024, kernel_initializer='glorot_normal'))
    model.add(LeakyReLU())

    model.add(Dense(1024, kernel_initializer='glorot_normal'))
    model.add(LeakyReLU())

    model.add(Dense(1024, kernel_initializer='glorot_normal'))
    model.add(LeakyReLU())

    model.add(Dense(1024, kernel_initializer='glorot_normal'))
    model.add(LeakyReLU())

    model.add(Dense(1024, kernel_initializer='glorot_normal'))
    model.add(LeakyReLU())

    model.add(Dense(1024, kernel_initializer='glorot_normal'))
    model.add(LeakyReLU())

    model.add(Dense(1024, kernel_initializer='glorot_normal'))
    model.add(LeakyReLU())

    model.add(Dense(1024, kernel_initializer='glorot_normal'))
    model.add(LeakyReLU())

    model.add(Dense(1024, kernel_initializer='glorot_normal'))
    model.add(LeakyReLU())
    model.add(Dense(units=7, activation='softmax', kernel_initializer='glorot_normal'))
    #using adamax
    model.compile(loss='categorical_crossentropy', optimizer="adamax", metrics=['accuracy'])
    model.summary()

#keras.callbacks.ReduceLROnPlateau
lrate = ReduceLROnPlateau(monitor='val_acc', factor=0.1, patience=4, verbose=1, mode='auto', epsilon=0.0001, cooldown=0, min_lr=0.00001)
#keras.callbacks.ModelCheckpoint
checkpoint = ModelCheckpoint(filepath=save_model_name+'.h5', monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
#keras.callbacks.EarlyStopping
earlystopping = EarlyStopping(monitor='val_acc', patience=20, verbose=1, mode='auto')


#data augmentation, fit with using image data generator
generate = ImageDataGenerator(rotation_range=30, width_shift_range=0.2, height_shift_range=0.2, zoom_range=[0.8, 1.2], shear_range=0.2, horizontal_flip=True)

generate.fit(X_train)
train_history = model.fit_generator(generate.flow(X_train, Y_train, batch_size=128), validation_data=(X_train_valid, Y_train_valid),
                            steps_per_epoch=len(X_train)/5, epochs=25, verbose=1, callbacks=[lrate,checkpoint,earlystopping])

'''
train_history = model.fit(x=X_train, y=Y_train,
                            #validation_data=(X_train, Y_train),
                            validation_data=(X_train_valid, Y_train_valid),
                            epochs=25, verbose=1, 
                            callbacks=[lrate,checkpoint,earlystopping]
                            )
'''
#save model
model.save(save_model_name+'.h5')

#savve history
with open(save_model_name+'_history','w') as f:
    f.write(str(train_history.history))

#show pic
show_train_history(train_history, 'acc', 'val_acc')
show_train_history(train_history, 'loss', 'val_loss')





