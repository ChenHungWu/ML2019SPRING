import numpy as np
import sys
import pandas as pd
from keras.models import Sequential
from keras.layers import AveragePooling2D, Dense, Dropout, Conv2D, MaxPooling2D, BatchNormalization, LeakyReLU, DepthwiseConv2D, Flatten
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.advanced_activations import LeakyReLU
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint, CSVLogger
seed = 40666888
pic_size = 48
X_train_address = sys.argv[1]
save_model_name = sys.argv[2]
def read_dataset(n):
    if n != 1:
        return np.load('X_train.npy'), np.load('Y_probability.npy')
    dataset = pd.read_csv(X_train_address)
    num_data = len(dataset)
    X_train = np.zeros((num_data,(48*48)))
    for i in range(num_data):
        X_train[i] =np.array(dataset['feature'][i].split())
    X_train = X_train.reshape(num_data,48,48,1)/255
    Y_train = np_utils.to_categorical(dataset['label'], 7)
    return X_train, Y_train
X_train, Y_train = read_dataset(1)
probability = pd.read_csv('label_probability.csv')
probability = np.array(probability.iloc[:,1:])
from sklearn.model_selection import train_test_split
X_train, X_train_valid, Y_train, Y_train_valid = train_test_split(X_train, probability, test_size=0.15, random_state=seed)
model = Sequential()
model.add(Conv2D(84,(3,3),input_shape = (48,48,1), padding='same', kernel_initializer='glorot_normal'))
model.add(BatchNormalization())
model.add(LeakyReLU())
model.add(MaxPooling2D(pool_size=(2,2), padding='same'))
model.add(DepthwiseConv2D(kernel_size=(2, 2), padding='same'))
model.add(BatchNormalization())
model.add(LeakyReLU())
model.add(Conv2D(84,(1,1), padding='same',kernel_initializer='glorot_normal'))
model.add(BatchNormalization())
model.add(LeakyReLU())
model.add(MaxPooling2D(pool_size=(2,2), padding='same'))
model.add(DepthwiseConv2D(kernel_size=(3, 3), padding='same'))
model.add(BatchNormalization())
model.add(LeakyReLU())
model.add(Conv2D(102,(1,1), padding='same',kernel_initializer='glorot_normal'))
model.add(BatchNormalization())
model.add(LeakyReLU())
model.add(MaxPooling2D(pool_size=(2,2), padding='same'))
model.add(DepthwiseConv2D(kernel_size=(3, 3), padding='same',))
model.add(BatchNormalization())
model.add(LeakyReLU())
model.add(Conv2D(102,(1,1), padding='same',kernel_initializer='glorot_normal'))
model.add(BatchNormalization())
model.add(LeakyReLU())
model.add(MaxPooling2D(pool_size=(2,2), padding='same'))
model.add(Flatten())
model.add(Dense(32, kernel_initializer='glorot_normal'))
model.add(BatchNormalization())
model.add(LeakyReLU())
model.add(Dense(32, kernel_initializer='glorot_normal'))
model.add(BatchNormalization())
model.add(LeakyReLU())
model.add(Dense(units=7, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='Nadam', metrics=['accuracy'])
model.summary()
csvlogger = CSVLogger('log_'+save_model_name+'.csv', append=False)
learning_rate = ReduceLROnPlateau(monitor='val_acc', factor=0.1, patience=4, verbose=1, mode='auto', epsilon=0.0001, cooldown=0, min_lr=0.00001)
checkpoint = ModelCheckpoint(filepath=save_model_name+'.weight', monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=True, mode='auto', period=1)
earlystopping = EarlyStopping(monitor='val_acc', patience=10, verbose=1, mode='auto', min_delta=0.0001)
model_json = model.to_json()
with open(save_model_name+'.json', "w") as json_file:
    json_file.write(model_json)
generate = ImageDataGenerator(rotation_range=15, width_shift_range=0.2, height_shift_range=0.2
    , zoom_range=[0.8, 1.2], shear_range=0.2, horizontal_flip=True)
generate.fit(X_train)
model.fit_generator(generate.flow(X_train, Y_train, batch_size=64),
    steps_per_epoch=len(X_train)/5, epochs=1000, verbose=1,
    validation_data=(X_train_valid, Y_train_valid),
    callbacks=[csvlogger, learning_rate, checkpoint, earlystopping])
