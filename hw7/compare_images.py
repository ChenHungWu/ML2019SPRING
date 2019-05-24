import sys
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from keras.models import Model, load_model
from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

model_num = '00'
autoencoder_name = './model/autoencoder_model'+model_num+'.h5'
encoder_name = './model/encoder_model'+model_num+'.h5'
seed = 40666888
#np.set_printoptions(threshold=np.inf)

def ReadData():
#read dataset and tranform into a 40000*(32*32*3) array
    dataset = []
    for n in range(40000):
        img = np.asarray( Image.open('images/{:0>6d}.jpg'.format(n+1)) )
        dataset.append(img)
    return(np.asarray(dataset))

images_array = np.load('images.npy')
images_array = images_array.astype('float32') / 255

############## For prob 2.c ##############
#draw images
autoencoder_model = load_model(autoencoder_name)
processed_images = autoencoder_model.predict(images_array)
for start in range(0, 32, 8):
    plt.figure()
    for i in range(0, 8, 1):
        # display original
        ax = plt.subplot(2, 8, i + 1)
        plt.imshow(images_array[i+start])
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    
        ax = plt.subplot(2, 8, i + 1 + 8)
        plt.imshow(processed_images[i+start])
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        diff = round(mean_absolute_error(images_array[i+start].reshape(1,(32*32*3))
            , processed_images[i+start].reshape(1,(32*32*3)))/(1/255), 2)
        ax.title.set_text(diff)

    plt.show()

############## For prob 2.b ##############
visualization = np.load('visualization.npy').astype('float32') / 255
print(visualization.shape)

encoder_model = load_model(encoder_name)
encoded_images = encoder_model.predict(visualization)

pca = PCA(n_components=300, whiten=True, random_state=seed)
pca.fit(encoded_images)
PCA_data=pca.transform(encoded_images)
print('Shape After PCA: ',PCA_data.shape)
K_result = KMeans(n_clusters=2, max_iter=500, n_init=50, verbose=0, n_jobs=-1, random_state=seed).fit(PCA_data)
K_means_result = np.array(K_result.labels_)
count=0
for i in range(2500):
    if K_means_result[i] == 0:
        count+=1
for i in range(2500, 5000):
    if K_means_result[i] == 1:
        count+=1
if(count<2500):
    count=5000-count
print('Dimension:', 300,' Accuracy: ', count/5000)

#plot 2D
A_as_A = np.where(K_means_result[:2500]==0)
A_as_B = np.where(K_means_result[:2500]==1)
B_as_A = np.where(K_means_result[2500:]==0)
B_as_B = np.where(K_means_result[2500:]==1)

pca = PCA(n_components=2, whiten=True, random_state=seed)
pca.fit(encoded_images)
PCA_data=pca.transform(encoded_images)
print('Shape After PCA: ',PCA_data.shape)
plt.scatter(x=PCA_data[:2500,0], y=PCA_data[:2500,1], color='red', label='DatasetA')
plt.scatter(x=PCA_data[2500:,0], y=PCA_data[2500:,1], color='blue', label='DatasetB') 
plt.legend(loc=1)
plt.show()

plt.scatter(x=PCA_data[A_as_A[0],0], y=PCA_data[A_as_A[0],1], color='red', label='DatasetA as A')
plt.scatter(x=PCA_data[B_as_B[0]+2500,0], y=PCA_data[B_as_B[0]+2500,1], color='blue', label='DatasetB as B') 
plt.scatter(x=PCA_data[A_as_B[0],0], y=PCA_data[A_as_B[0],1], color='black', label='DatasetA as B')
plt.scatter(x=PCA_data[B_as_A[0]+2500,0], y=PCA_data[B_as_A[0]+2500,1], color='gold', label='DatasetB as A')
plt.legend(loc=1)
plt.show()
