import numpy as np
import sys
from lime import lime_image
from skimage.segmentation import slic
import matplotlib.pyplot as plt
from keras.models import Model, load_model
# for segmentation function
from lime.wrappers.scikit_image import SegmentationAlgorithm
# for masking boundary
from skimage.segmentation import mark_boundaries
import skimage

input_path = sys.argv[1]
output_path = sys.argv[2]
pic_size = 48
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

# number of class
class_num = 7 

def predicFunction(X_train) :
    X_train = skimage.color.rgb2gray(X_train)
    X_train = X_train.reshape(-1,48,48,1)
    return model.predict(X_train)

def segmentation(X_train):
    np.random.seed(666)
    return skimage.segmentation.slic(image=X_train)

# loading previous model
model_name = 'en05.h5'
model = load_model(model_name)

# loading data for explanation
X_train, Y_train_label = from_dataset()
#X_train = np.load('X_train.npy')
#Y_train_label = np.load('Y_train_label.npy')
#predict = model.predict(X_train)

class_num = 7
classes = ["Angry","Disgust","Fear","Happy","Sad","Surprise","Neutral"] #list of class

pic_list = []#the pic_num to be show
for c in range(class_num):
	pic_list.append([])
	for i in range(len(Y_train_label)):
		if(Y_train_label[i]==c 
			#and np.argmax(predict[i])==c
			): 
			pic_list[c].append(i)

for i in range(class_num):#find one pic_num of each class
	pic_list[i] = pic_list[i][0]

for i in range(len(pic_list)):
	img = X_train[pic_list[i]]
	img = np.reshape(img,(48,48))
	
	img = skimage.color.gray2rgb(img)

	# Initiate explainer instance
	explainer = lime_image.LimeImageExplainer()

	# Get the explaination of an image
	explanation = explainer.explain_instance(
		image=img, 
		classifier_fn=predicFunction,
		segmentation_fn=segmentation,
		hide_color = 0,
		num_samples=2000,
		top_labels=10)
		
	# Get processed image
	image, mask = explanation.get_image_and_mask(
		label=Y_train_label[i],
		positive_only=False,
		hide_rest=False,
		num_features=10,
		min_weight=0.0)

	# save and plot the image
	plt.title(classes[i])
	plt.xticks([])
	plt.yticks([])
	plt.imshow(mark_boundaries(skimage.color.gray2rgb(image.reshape(48,48,3)), mask),interpolation ='nearest')
	plt.savefig(output_path+'fig3_'+str(i)+'.jpg')
	#plt.show()


