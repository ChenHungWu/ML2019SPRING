import sys
import numpy as np
from PIL import Image
from keras.optimizers import RMSprop
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, BatchNormalization
from keras.layers import Conv2DTranspose, Activation, Reshape, Flatten
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint, CSVLogger
from keras.models import Model, load_model
from keras import backend as K
#https://codertw.com/%E7%A8%8B%E5%BC%8F%E8%AA%9E%E8%A8%80/568066/#outline__3
#https://blog.keras.io/building-autoencoders-in-keras.html
#https://github.com/ardamavi/Unsupervised-Classification-with-Autoencoder/blob/master/Unsupervised%20Classification%20With%20Autoencoder.ipynb
model_num = '16'
autoencoder_name = './model/autoencoder_model'+model_num
encoder_name = './model/encoder_model'+model_num
seed = 40666888

def ReadData():
#read dataset and tranform into a 40000*(32*32*3) array
    dataset = []
    for n in range(40000):
        img = np.asarray( Image.open('images/{:0>6d}.jpg'.format(n+1)) )
        dataset.append(img)
    return(np.asarray(dataset))


#data = ReadData()
#data = data.reshape(40000,32,32,1)
#np.save('images_gray.npy',data)

images_array = np.load('images.npy')
images_array = images_array.astype('float32') / 255

#Encoder:
input_img = Input(shape=(32, 32, 3))
conv_1 = Conv2D(128, (3,3), strides=(1,1))(input_img)
bn_1 = BatchNormalization()(conv_1)
act_1 = Activation('relu')(bn_1)
maxpool_1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(act_1)

conv_2 = Conv2D(128, (3,3), strides=(1,1), padding='same')(maxpool_1)
bn_2 = BatchNormalization()(conv_2)
act_2 = Activation('relu')(bn_2)
maxpool_2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(act_2)

flat_1 = Flatten()(maxpool_2)

fc_1 = Dense(4096)(flat_1)
bn_3 = BatchNormalization()(fc_1)
act_3 = Activation('relu')(bn_3)

#Decoder:
fc_5 = Dense(3136*2)(act_3)
bn_4 = BatchNormalization()(fc_5)
act_6 = Activation('relu')(bn_4)

reshape_1 = Reshape((7,7,128))(act_6)

upsample_1 = UpSampling2D((2, 2))(reshape_1)
deconv_1 = Conv2DTranspose(128, (3, 3), strides=(1, 1))(upsample_1)
bn_5 = BatchNormalization()(deconv_1)
act_7 = Activation('relu')(bn_5)

upsample_2 = UpSampling2D((2, 2))(act_7)
deconv_2 = Conv2DTranspose(128, (3, 3), strides=(1, 1))(upsample_2)
bn_6 = BatchNormalization()(deconv_2)
act_8 = Activation('relu')(bn_6)

conv_3 = Conv2D(3, (3, 3), strides=(1, 1))(act_8)
bn_7 = BatchNormalization()(conv_3)
act_9 = Activation('sigmoid')(bn_7)

autoencoder = Model(input_img, act_9)
autoencoder.compile(optimizer='rmsprop', loss='mae')
autoencoder.summary()

#from sklearn.model_selection import train_test_split
#train_X,valid_X,train_ground,valid_ground = train_test_split(images_array, images_array, test_size=0.05, random_state=seed)

checkpoint = ModelCheckpoint(monitor='loss', filepath=(autoencoder_name+'.h5'), verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
learning_rate = ReduceLROnPlateau(monitor='loss', patience=3, factor=0.3, verbose=1, mode='auto', epsilon=0.0002, cooldown=0, min_lr=0.000005)
earlystopping = EarlyStopping(monitor='loss', patience=8, verbose=1, mode='auto', min_delta=0.0002)
csvlogger = CSVLogger(('./train_log/log_autoencoder_model'+model_num+'.csv'), append=False)

#from keras.preprocessing.image import ImageDataGenerator
#generated_data = ImageDataGenerator(featurewise_center=False, samplewise_center=False
#	, featurewise_std_normalization=False, samplewise_std_normalization=False
#	, zca_whitening=False, rotation_range=0,  width_shift_range=0.1, height_shift_range=0.1
#	, horizontal_flip = True, vertical_flip = False)
#generated_data.fit(train_X)
#autoencoder.fit_generator(generated_data.flow(train_X, train_ground, batch_size=256), steps_per_epoch=train_X.shape[0]/5,
#  epochs=50, validation_data=(valid_X, valid_ground), shuffle=True, verbose=1,
#  callbacks=[learning_rate, checkpoint, earlystopping, csvlogger])

autoencoder.fit(images_array, images_array, batch_size=64, epochs=5000, shuffle=True
     , verbose=1, callbacks=[learning_rate, checkpoint, earlystopping, csvlogger])

encoder = Model(input_img, act_3)
#encoder.summary()
encoder.save(encoder_name+'.h5')