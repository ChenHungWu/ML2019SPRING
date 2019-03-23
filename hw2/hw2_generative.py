import sys
import numpy as np
train_data = sys.argv[1]
test_data = sys.argv[2]
X_train = sys.argv[3]
Y_train = sys.argv[4]
X_test = sys.argv[5]
output_name = sys.argv[6]

#Read Train Data
x_train = np.genfromtxt(X_train, delimiter = ',',skip_header = 1)
y_train = np.genfromtxt(Y_train, delimiter = ',',skip_header = 1)
y_train = np.reshape(y_train,(len(y_train),1))
f_size = len(x_train[0]) # feature size
d_size = len(x_train) # data size


std = np.std(x_train,axis=0)
mean = np.mean(x_train,axis=0)
x_train = np.divide(np.subtract(x_train,mean),std)

#Define Sigmoid Function
def sigmoid(z):
    return np.clip((1/(1+np.exp(-z))), 0.00000000000001, 0.99999999999999)

#Find Mu
num_of_1 = 0
num_of_0 = 0
mu_of_1 = np.zeros((1,f_size))
mu_of_0 = np.zeros((1,f_size))
for n in range(d_size):
	if y_train[n] == 1:
		mu_of_1 += x_train[n]
		num_of_1 += 1
	else:
		mu_of_0 += x_train[n]
		num_of_0 += 1
mu_of_1 /= num_of_1
mu_of_0 /= num_of_0

#Find Sigma
sig_of_1 = np.zeros((f_size,f_size))
sig_of_0 = np.zeros((f_size,f_size))
for n in range(d_size):
	if y_train[n] == 1:
		sig_of_1 += np.dot(np.transpose(x_train[n]-mu_of_1),(x_train[n]-mu_of_1))
	else:
		sig_of_0 += np.dot(np.transpose(x_train[n]-mu_of_0),(x_train[n]-mu_of_0))
sig_of_1 /= num_of_1
sig_of_0 /= num_of_0
sig = float(num_of_1)/d_size*sig_of_1 + float(num_of_0)/d_size*sig_of_0 # share sigma
inverse_sig = np.linalg.inv(sig)

#Get w and b
w = np.dot(mu_of_1-mu_of_0,inverse_sig)
b = np.log(float(num_of_1)/num_of_0)-0.5*np.dot(np.dot(mu_of_1,inverse_sig),np.transpose(mu_of_1))+0.5*np.dot(np.dot(mu_of_0,inverse_sig),np.transpose(mu_of_0))
	
#Test
x_test = np.genfromtxt(X_test, delimiter = ',',skip_header = 1)
x_test = np.divide(np.subtract(x_test,mean),std)
predict = np.around(sigmoid(np.dot(x_test,np.transpose(w))+b))
save = open(output_name, 'w')
save.write("id,label\n")
for i in range(len(predict)):
    save.write(str(i+1) + "," + str(int(predict[i])) + "\n")
save.close()
