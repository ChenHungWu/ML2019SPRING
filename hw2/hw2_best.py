import sys
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier

train_data = sys.argv[1]
test_data = sys.argv[2]
X_train = sys.argv[3]
Y_train = sys.argv[4]
X_test = sys.argv[5]
output_name = sys.argv[6]

train_X = np.genfromtxt(X_train, delimiter = ',',skip_header = 1)
train_Y = np.genfromtxt(Y_train, delimiter = ',',skip_header = 1)

#normalize
std = np.std(train_X,axis=0)
mean = np.mean(train_X,axis=0)
train_X = np.divide(np.subtract(train_X,mean),std)

test_X = np.genfromtxt(X_test, delimiter = ',',skip_header = 1)
#normalize
test_X = np.divide(np.subtract(test_X,mean),std)

clf = GradientBoostingClassifier(learning_rate=0.26,subsample=0.65,random_state=10,n_estimators=70,max_depth=3,min_samples_split=8)
clf.fit(train_X, train_Y)
predict_ans = clf.predict(test_X)

save = open(output_name,'w')
save.write("id,label\n")
for i in range(len(test_X)):
    save.write(str(i+1) + "," + str(int(predict_ans[i])) + "\n")
save.close()



