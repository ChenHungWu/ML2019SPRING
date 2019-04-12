import sys
import numpy as np, random
import matplotlib.pyplot as plt
from keras.models import load_model
from keras import backend as K
#https://www.kaggle.com/ernie55ernie/mnist-with-keras-visualization-and-saliency-map
#https://github.com/WindQAQ/ML2017/blob/master/hw3/saliency_map.py

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
	return X_train, Y_train_label

#X_train, Y_train_label = from_dataset()
X_train , Y_train_label= np.load('X_train.npy') ,np.load('Y_train_label.npy') #before One-hot encoding
class_num = 7
classes = ["Angry","Disgust","Fear","Happy","Sad","Surprise","Neutral"] #list of class

pic_list = []#the pic_num to be show

for i in range(class_num):#find one pic_num of each class
    pic_list.append((np.where(Y_train_label==i))[0][0])

model = load_model(model_name)
input_tensors = [model.input]

for i in range(len(pic_list)):
    fig, ax = plt.subplots(nrows=3, ncols=1, figsize = (6,6))
    fig.suptitle('Saliency_map')

    #plot origin pic
    img = X_train[pic_list[i]]
    ax[0].set_title(classes[i])
    cax = ax[0].imshow(img.reshape((48, 48))*255, cmap = 'gray')
    fig.colorbar(cax, ax = ax[0], orientation='vertical',fraction=0.046, pad=0.04)
    ax[0].set_xticks([])
    ax[0].set_yticks([])

    #plot heat map
    gradients = model.optimizer.get_gradients(model.output[0][ int( Y_train_label[ pic_list[i] ] ) ], model.input)
    compute_gradients = K.function(inputs = input_tensors, outputs = gradients)
    x_value = np.expand_dims(img, axis=0)
    val_grads = compute_gradients([x_value])[0][0]
    #val_grads = np.maximum(val_grads, 0)
    val_grads /= np.max(val_grads)
    heatmap = val_grads.reshape(48, 48)
    ax[1].set_title('Heat map')
    cax = ax[1].imshow(heatmap, cmap = 'jet')
    fig.colorbar(cax, ax = ax[1], orientation='vertical',fraction=0.046, pad=0.04)
    ax[1].set_xticks([])
    ax[1].set_yticks([])

    #plot the model focus on
    thres = 0.1
    see = img.reshape(48, 48)
    see[np.where(heatmap <= thres)] = np.mean(see)
    ax[2].set_title('Masked')
    cax = ax[2].imshow(see*255, cmap = 'gray')
    fig.colorbar(cax, ax = ax[2], orientation='vertical',fraction=0.046, pad=0.04)
    ax[2].set_xticks([])
    ax[2].set_yticks([])
    plt.savefig(output_path+'fig1_'+str(i)+'.jpg')
    #plt.show()
