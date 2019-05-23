import sys
import numpy as np
import pandas as pd
from PIL import Image
from keras.models import load_model
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn import cluster

img_address = sys.argv[1]
test_data_address = sys.argv[2]
ans_address = sys.argv[3]

encoder_name = './encoder_model.h5'
seed = 40666888
pic_num = 40000
dim_list = [1400, 1700, 2000]


def ReadData():
#read dataset and tranform into a 40000*(32*32*3) array
    dataset = []
    for n in range(pic_num):
        img = np.asarray( Image.open( img_address+'{:0>6d}.jpg'.format(n+1)) )
        dataset.append(img)
    return(np.asarray(dataset))

images_array = ReadData()
images_array = images_array.astype('float32') / 255

encoder_model = load_model(encoder_name)
encoded_images = encoder_model.predict(images_array)

print('Shape Before PCA: ',encoded_images.shape)

record = pd.DataFrame()
for i in dim_list:
    pca = PCA(n_components=i, whiten=True, random_state=seed)
    pca.fit(encoded_images)
    PCA_data=pca.transform(encoded_images)
    print('Shape After PCA: ',PCA_data.shape)

    K_result = KMeans(n_clusters=2, max_iter=500, n_init=50, verbose=0, n_jobs=-1, random_state=seed).fit(PCA_data)
    K_means_result = np.array(K_result.labels_)

    record.insert(len(record.columns), str(i), K_means_result)

first_column = record.columns[0]

for i in range(1,len(record.columns)):
    compare_name = record.columns[i]
    count = 0
    for j in range(pic_num):
        if record[first_column][j] == record[compare_name][j]:
            count+=1
    accuracy = count/pic_num
    print(accuracy)
    if(accuracy<0.5):
        accuracy = 1-accuracy
        for member in range(pic_num):
            if record[compare_name][member] == 0:
                record[compare_name][member] = 1
            else:
                record[compare_name][member] = 0


testFile = pd.read_csv(test_data_address)
index1, index2 = testFile['image1_name'], testFile['image2_name']

ensemble = pd.DataFrame()
for i in record.columns:
    ensemble += record[i]

for i in range(pic_num):
    if ensemble[i]>=2:
        ensemble[i]=1
    else :
        ensemble[i]=0

result = []
for i in range( len(testFile) ) :
    if ensemble[ index1[i]-1 ] == ensemble[ index2[i]-1 ] :
        result.append(1)
    else :
        result.append(0)

predict_result = pd.DataFrame(result)
predict_result.index.name='id'
predict_result.to_csv(ans_address, index=True, header=['label'])



