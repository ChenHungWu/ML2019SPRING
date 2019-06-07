import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Activation, BatchNormalization
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.advanced_activations import LeakyReLU
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint, CSVLogger

myseed = 10
drop_rate = 0.25
pic_size = 48

'''Addresses'''
X_train_address = sys.argv[1]
save_model_name = "cnn"

'''Function Define'''
def read_dataset():
    #from_dataset
    dataset = pd.read_csv(X_train_address)
    num_data = len(dataset)
    X_train = np.zeros((num_data,(48*48)))
    for i in range(num_data):
        X_train[i] =np.array(dataset['feature'][i].split())
    X_train = X_train.reshape(num_data,48,48,1)/255

    Y_train = np_utils.to_categorical(dataset['label'], 7)

    return X_train, Y_train

'''
def show_train_history(train_history, train, validation):
    plt.plot(train_history.history[train])
    plt.plot(train_history.history[validation])
    plt.title('Train History')
    plt.ylabel('train')
    plt.xlabel('Epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()
'''

#read data
X_train, Y_train = read_dataset()

#cut valid set
from sklearn.model_selection import train_test_split
X_train, X_train_valid, Y_train, Y_train_valid = train_test_split(X_train, Y_train, test_size=0.15, random_state=seed)

#build model
if 1+1 == 2:

    model = Sequential()
    
    model.add(Conv2D(filters=512, kernel_size=(3,3), input_shape=(pic_size,pic_size,1), padding='same', kernel_initializer='glorot_normal'))
    model.add(LeakyReLU())
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2), padding='same'))
    model.add(Dropout(rate=drop_rate, seed=myseed))

    model.add(Conv2D(filters=512, kernel_size=(3,3), padding='same', kernel_initializer='glorot_normal'))
    model.add(LeakyReLU())
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2), padding='same'))
    model.add(Dropout(rate=drop_rate, seed=myseed))

    model.add(Conv2D(filters=768, kernel_size=(3,3), padding='same', kernel_initializer='glorot_normal'))
    model.add(LeakyReLU())
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2), padding='same'))
    model.add(Dropout(rate=drop_rate, seed=myseed))

    model.add(Conv2D(filters=768, kernel_size=(3,3), padding='same', kernel_initializer='glorot_normal'))
    model.add(LeakyReLU())
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2), padding='same'))
    model.add(Dropout(rate=drop_rate, seed=myseed))
    
    model.add(Flatten())

    model.add(Dense(512, kernel_initializer='glorot_normal'))
    model.add(LeakyReLU())
    model.add(BatchNormalization())
    model.add(Dropout(rate=drop_rate, seed=myseed))

    model.add(Dense(512, kernel_initializer='glorot_normal'))
    model.add(LeakyReLU())
    model.add(BatchNormalization())
    model.add(Dropout(rate=drop_rate, seed=myseed))

    model.add(Dense(units=7, activation='softmax', kernel_initializer='glorot_normal'))
    #using adamax
    model.compile(loss='categorical_crossentropy', optimizer="adamax", metrics=['accuracy'])
    model.summary()

lrate = ReduceLROnPlateau(monitor='val_acc', factor=0.1, patience=4, verbose=1, mode='auto', epsilon=0.0001, cooldown=0, min_lr=0.00001)
checkpoint = ModelCheckpoint(filepath=save_model_name+'.h5', monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
earlystopping = EarlyStopping(monitor='val_acc', patience=20, verbose=1, mode='auto')
csvlogger = CSVLogger('log_'+save_model_name+'.csv', append=False)

#data augmentation, fit with using image data generator
generate = ImageDataGenerator(rotation_range=30, width_shift_range=0.2, height_shift_range=0.2, zoom_range=[0.8, 1.2], shear_range=0.2, horizontal_flip=True)

generate.fit(X_train)
train_history = model.fit_generator(generate.flow(X_train, Y_train, batch_size=128), validation_data=(X_train_valid, Y_train_valid),
                            steps_per_epoch=len(X_train)/5, epochs=25, verbose=1, callbacks=[lrate,checkpoint,earlystopping,csvlogger])

'''
train_history = model.fit(x=X_train, y=Y_train,
                            #validation_data=(X_train, Y_train),
                            validation_data=(X_train_valid, Y_train_valid),
                            epochs=25, verbose=1, 
                            callbacks=[lrate,checkpoint,earlystopping,csvlogger]
                            )
'''

#show pic
#show_train_history(train_history, 'acc', 'val_acc')
#show_train_history(train_history, 'loss', 'val_loss')





