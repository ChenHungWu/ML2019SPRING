import sys
import numpy as np
import keras.models
from keras.models import load_model
pic_size = 48

ad_testset = sys.argv[1]
ad_export = sys.argv[2]
#model_name = 'cnn.h5'
def from_dataset():
	#load in by genfromtxt
	dataset = np.genfromtxt(fname=ad_testset, dtype='str', delimiter=' ', skip_header=1)
	dataset = dataset.T
	
	#fix data into label & feature
	fixdata = dataset[0]
	label = np.zeros(len(fixdata))
	fix = np.zeros(len(fixdata))
	for n in range(len(fixdata)):
		label[n], fix[n] = fixdata[n].split(',')[0], fixdata[n].split(',')[1]
	dataset[0] = fix

	feature = np.reshape(dataset.T,(7178,pic_size,pic_size,1)).astype('float64')
	feature /= 255
	'''
	mean_std = np.load('mean_std.npy').astype('float64')
	feature = np.divide(np.subtract(feature,mean_std[0]),(mean_std[1] + 1e-10))
	'''
	#save feature and label as npy
	#np.save('testfeature.npy',feature)
	return feature

def export(label, address):
	save = open(address,'w')
	save.write("id,label\n")
	for i in range(len(label)):
		save.write(str(i)+","+str(int(np.argmax(label[i])))+"\n")
	save.close()

def from_npy():
	return np.load('testfeature.npy')

def read_testset(n):
	if n == 1:
		return from_dataset()
	else:
		return from_npy()

'''Main'''
testfeature = read_testset(1)

#model00 = load_model(model_name)
model01 = load_model('en01.h5')
model02 = load_model('en02.h5')
model03 = load_model('en03.h5')
model04 = load_model('en04.h5')
model05 = load_model('en05.h5')
model06 = load_model('en06.h5')
model07 = load_model('en07.h5')


#testlabel00 = model00.predict(testfeature)
testlabel01 = model01.predict(testfeature)
testlabel02 = model02.predict(testfeature)
testlabel03 = model03.predict(testfeature)
testlabel04 = model04.predict(testfeature)
testlabel05 = model05.predict(testfeature)
testlabel06 = model06.predict(testfeature)
testlabel07 = model07.predict(testfeature)

testlabel_total =  (testlabel01 + testlabel02 + testlabel03 + testlabel04 + testlabel05)/5
testlabel_total += testlabel06 + testlabel07
testlabel_total = testlabel00
save = open(ad_export,'w')
save.write("id,label\n")
for i in range(len(testlabel_total)):
	save.write(str(i)+","+str(int(np.argmax(testlabel_total[i])))+"\n")
save.close()