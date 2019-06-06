import sys 
import numpy as np
import pandas as pd
from sklearn.externals import joblib
X_test_address = sys.argv[1]
result_file = sys.argv[2]
model_name = sys.argv[3]
dataset = pd.read_csv(X_test_address)
X_test = np.zeros((7178,(48*48)))
for i in range(7178):
    X_test[i] =np.array(dataset['feature'][i].split())
X_test = X_test.reshape(7178,48,48,1)/255
model = joblib.load(model_name)
Y_preds = model.predict(X_test)
f = open(result_file,'w')
f.write("id,label\n")
for i in range(len(Y_preds)) :
    f.write(str(i) + ',' + str(int(np.argmax(Y_preds[i]))) + '\n')
f.close()

