import sys
import numpy as np
from PIL import Image
import keras
from keras.applications import vgg16, vgg19, resnet50, densenet
from keras_applications.resnet import ResNet101

pic_num = 200
start = 0
original_pic_address = sys.argv[1]
attack_address = sys.argv[2]
label_address = sys.argv[3]

def cal_L_infinity(ground_truth, test_data):
	diff = np.abs(ground_truth - test_data)
	max_diff = np.zeros(len(ground_truth))
	for i in range(len(ground_truth)):
		max_diff[i] = np.max(diff[i])
	print(max_diff)
	return np.average(max_diff)
'''
labels = np.loadtxt("labels.csv", dtype=np.str, delimiter=',') 
labels = labels[1:]
labels = labels[:,3]
np.save('labels.npy', labels)
'''
labels = np.load(label_address) 
labels = labels[start:start+pic_num]

img_array = np.zeros((pic_num,224,224,3))
for i in range(start,start+pic_num):
    img = Image.open(original_pic_address+'/{:0>3d}.png'.format(i))#input
    img_array[i%pic_num] = np.array(img)

pic_array = np.zeros((pic_num,224,224,3))
for i in range(start,start+pic_num):
    img = Image.open(attack_address+'/{:0>3d}.png'.format(i))#input
    pic_array[i%pic_num] = np.array(img)
print('L-infinity = ',cal_L_infinity(img_array[start:start+pic_num],pic_array[start:start+pic_num]))

#vgg16
model = vgg16.VGG16(weights='imagenet')
preprocessed_images = vgg16.preprocess_input(pic_array)
preds = model.predict(preprocessed_images)
preds = np.argmax(preds,axis=1)	
count = 0
for i in range(pic_num):
	if( int(labels[i]) != preds[i]):
		count = count+1
print('Vgg-16 prediction attack rate:',count/pic_num)	

#vgg19
model = vgg19.VGG19(weights='imagenet')
for i in range(start,start+pic_num):
    img = Image.open(attack_address+'/{:0>3d}.png'.format(i))#input
    pic_array[i%pic_num] = np.array(img)
preprocessed_images = vgg19.preprocess_input(pic_array)
preds = model.predict(preprocessed_images)
preds = np.argmax(preds,axis=1)	
count = 0
for i in range(pic_num):
	if( int(labels[i]) != preds[i]):
		count = count+1
print('Vgg-19 prediction attack rate:',count/pic_num)	

#ResNet-50
model = resnet50.ResNet50(weights='imagenet')
for i in range(start,start+pic_num):
    img = Image.open(attack_address+'/{:0>3d}.png'.format(i))#input
    pic_array[i%pic_num] = np.array(img)
preprocessed_images = resnet50.preprocess_input(pic_array)
preds = model.predict(preprocessed_images)
preds = np.argmax(preds,axis=1)	
count = 0
for i in range(pic_num):
	if( int(labels[i]) != preds[i]):
		count = count+1
print('ResNet-50 prediction attack rate:',count/pic_num)

#ResNet-101
model = ResNet101(weights='imagenet', backend = keras.backend, layers = keras.layers, models = keras.models, utils  = keras.utils )
for i in range(start,start+pic_num):
    img = Image.open(attack_address+'/{:0>3d}.png'.format(i))#input
    pic_array[i%pic_num] = np.array(img)
preprocessed_images = resnet50.preprocess_input(pic_array)
preds = model.predict(preprocessed_images)
preds = np.argmax(preds,axis=1)	
count = 0
for i in range(pic_num):
	if( int(labels[i]) != preds[i]):
		count = count+1
print('ResNet-101 prediction attack rate:',count/pic_num)	


#DenseNet121
model = densenet.DenseNet121(weights='imagenet')
for i in range(start,start+pic_num):
    img = Image.open(attack_address+'/{:0>3d}.png'.format(i))#input
    pic_array[i%pic_num] = np.array(img)
preprocessed_images = densenet.preprocess_input(pic_array)
preds = model.predict(preprocessed_images)
preds = np.argmax(preds,axis=1)	
count = 0
for i in range(pic_num):
	if( int(labels[i]) != preds[i]):
		count = count+1
print('DenseNet-121 prediction attack rate:',count/pic_num)

#DenseNet169
model = densenet.DenseNet169(weights='imagenet')
for i in range(start,start+pic_num):
    img = Image.open(attack_address+'/{:0>3d}.png'.format(i))#input
    pic_array[i%pic_num] = np.array(img)
preprocessed_images = densenet.preprocess_input(pic_array)
preds = model.predict(preprocessed_images)
preds = np.argmax(preds,axis=1)	
count = 0
for i in range(pic_num):
	if( int(labels[i]) != preds[i]):
		count = count+1
print('Densenet-169 prediction attack rate:',count/pic_num)	