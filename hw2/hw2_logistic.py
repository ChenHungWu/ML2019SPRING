import sys
import numpy as np
import matplotlib.pyplot as plt

train_data = sys.argv[1]
test_data = sys.argv[2]
X_train = sys.argv[3]
Y_train = sys.argv[4]
X_test = sys.argv[5]
output_name = sys.argv[6]

degree = int(7)

'''
x_name = np.loadtxt(X_train, dtype=np.str, delimiter=",")
x_name = x_name[0]
for i in range(2,degree):
	for j in range(6):
		x_name = np.append(x_name,(x_name[j]+'_'+str(degree)))
x_name = np.append(x_name,'bias')
'''

#Read Train Data
x_train = np.genfromtxt(X_train, delimiter = ',',skip_header = 1)

for i in range(2,degree):
	x_train = np.concatenate((x_train,x_train[:,range(0,6)]**i), axis=1)

y_train = np.genfromtxt(Y_train, delimiter = ',',skip_header = 1)
y_train = np.reshape(y_train,(len(y_train),1))


#Define Sigmoid Function
def sigmoid(z):
    return np.clip((1/(1+np.exp(-z))), 0.00000000000001, 0.99999999999999)

#Normalize
std = np.std(x_train,axis=0)
mean = np.mean(x_train,axis=0)
x_train = np.divide(np.subtract(x_train,mean),std)
#add bias
x_train = np.concatenate((np.ones((x_train.shape[0],1)),x_train), axis=1)
size = len(x_train[0])

#Gradient Decent
w_mt = np.zeros((size,1))
w_vt = np.zeros((size,1))
b1 = 0.9
b2 = 0.999
e = 1e-8
w = np.full((size,1),0.1)
l_rate = 0.01
iteration = 50000

w_mt = np.load('w_mt.npy')
w_vt = np.load('w_vt.npy')
w = np.load('w.npy')

'''
#lamda = [0.001,0.003,0.005,0.007]
#color = ['red','green','blue','gray','peru','g','gold']
#Start Training
#for t in range(len(lamda)):
w = np.full((size,1),0.1)
w_mt = np.zeros((size,1))
w_vt = np.zeros((size,1))
losslist = []
for it in range(1,iteration):
	est = sigmoid(np.dot(x_train,w))
	loss = y_train - est
	#regularization
	#loss += lamda[t] * np.sum( np.concatenate((np.zeros((1,1)),w[1:]),axis=0) **2 )
	w_grad = -2*np.dot(np.transpose(x_train),loss) # + 2*lamda[t]*(np.concatenate((np.zeros((1,1)),w[1:]),axis=0))
	print(np.sum(-(y_train*np.log(est)+(1-y_train)*np.log(1-est))))
	#s = np.sum(loss**2)/len(loss)
	#losslist.append(s)
	#print(s)
	#print(it,np.sum(-(y_train*np.log(est)+(1-y_train)*np.log(1-est))),'%')
	w_mt = b1*w_mt + (1-b1)*w_grad
	w_vt = b2*w_vt + (1-b2)*(w_grad*w_grad)
	wm_cap = w_mt / (1-(b1**it))
	wv_cap = w_vt / (1-(b2**it))
	w = w - (l_rate * wm_cap) / (np.sqrt(wv_cap)+e) 
#plt.plot(losslist,color[t])
#plt.show()
'''

#Save Load

#np.save('w_mt.npy',w_mt)
#np.save('w_vt.npy',w_vt)
#np.save('w.npy',w)


#Test
'''
w_copy = w*1
mydic = {}
for i in range(w_copy.shape[0]):
	mydic[float(w_copy[i])] = int(i)
w_copy = np.sort(axis=0,a=w_copy)
w_copy = w_copy[::-1]

for i in range(w_copy.shape[0]):
	print (x_name[mydic[float(w_copy[i])]],w_copy[i])
'''

x_test = np.genfromtxt(X_test, delimiter = ',',skip_header = 1)

for i in range(2,degree):
	x_test = np.concatenate((x_test,x_test[:,range(0,6)]**i), axis=1)

x_test = np.divide(np.subtract(x_test,mean),std)
x_test = np.concatenate((np.ones((x_test.shape[0],1)),x_test), axis=1)
y_test = (sigmoid(np.dot(x_test,w))) >0.53

save = open(output_name, 'w')
save.write("id,label\n")
for i in range(len(y_test)):
    save.write(str(i+1) + "," + str(int(y_test[i])) + "\n")
save.close()

