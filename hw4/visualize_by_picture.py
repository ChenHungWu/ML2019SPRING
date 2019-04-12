import sys
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
from keras import backend as K

input_path = sys.argv[1]
output_path = sys.argv[2]
pic_size = 48
model_name = 'en05.h5'

def from_dataset():
	#load in by genfromtxt
	dataset = np.genfromtxt(fname=input_path, dtype='str', delimiter=' ', skip_header=1)
	num_data = len(dataset)
	dataset = dataset.T
	#fix data into Y_train_label & X_train
	fixdata = dataset[0]
	Y_train_label = np.zeros(len(fixdata))
	fix = np.zeros(len(fixdata))
	for n in range(len(fixdata)):
		Y_train_label[n], fix[n] = fixdata[n].split(',')[0], fixdata[n].split(',')[1]
	dataset[0] = fix
	
	X_train = np.reshape(dataset.T,(num_data,pic_size,pic_size,1)).astype('float64')
	X_train /= 255
	#One-hot encoding
	#Y_train = np_utils.to_categorical(Y_train, 7)
	return X_train
	
# loading previous model
model = load_model(model_name)

# loading data for explanation
X_train = from_dataset()
#X_train = np.load('X_train.npy')

name_ls = ['leaky_re_lu_1']
#name_ls = ['leaky_re_lu_1', 'leaky_re_lu_2', 'leaky_re_lu_3', 'leaky_re_lu_4']
#name_ls = ['conv2d_1', 'conv2d_2']

nb_filter = 32
image_idx = 0

#model.predict(X_train[0].reshape((1, 48, 48, 1)))

for cnt, c in enumerate(name_ls):
    #print('Process layer {}'.format(name_ls[cnt]))
    filter_imgs = []
    input_img_data = X_train[image_idx].reshape((1, 48, 48, 1))
    get_layer_output = K.function( [model.layers[0].input], [model.get_layer(c).output] )
    layer_output = get_layer_output([input_img_data])[0]
    filter_imgs.append([layer_output[:,:,:,asd] for asd in range(nb_filter)])

    #start plot
    fig , ax = plt.subplots( nrows= nb_filter//8, ncols = 8 , figsize=(14, 8))
    plt.tight_layout()
    for i in range(nb_filter):
        ax[i//8,i%8].imshow(layer_output[0,:,:,i].squeeze(), cmap='BuGn')
        ax[i//8,i%8].set_xticks([])
        ax[i//8,i%8].set_yticks([])
        ax[i//8,i%8].set_title('filter '+str(i)) 
    fig.savefig(output_path+'fig2_2.jpg')
