import numpy as np
from numpy.linalg import svd
from skimage import io
import sys, os
#http://www.cnblogs.com/LeftNotEasy/archive/2011/01/19/svd-and-applications.html
images_path = sys.argv[1] 
input_image = sys.argv[2] 
reconstruct_image = sys.argv[3] 

# Number of principal components used
k = 5

def process(M): 
    M -= np.min(M)
    M /= np.max(M)
    M = (M * 255).astype(np.uint8)
    return M

def reconstruction(images_array, eigen):
    ############ For prob 1.c ############
    picked_img = io.imread(os.path.join(images_path, input_image))
    test_image = picked_img.flatten().astype('float32') 

    # Calculate mean & Normalize
    mean =  np.mean(images_array, axis=0)
    test_image -= mean
    

    # Compression
    weight = np.dot(test_image, eigen)
    
    # Reconstruction
    recon = np.zeros(600*600*3)
    for i in range(k):
        layer = weight[i] * eigen[:,i]
        recon += layer
    recon = process(recon+mean).reshape(600,600,3)
    io.imsave(reconstruct_image, recon, quality=100)


def average_face(images_array):
	############ For prob 1.a ############
	average = np.mean(images_array, axis=0)
	average = process(average).reshape(600,600,3)
	io.imsave('average.jpg', average)

def eigen_face(images_array):

    images_array = images_array.astype(np.float64)
    average = np.mean(images_array, axis=0).astype(np.float64)
    images_array -= average

    # Use SVD to find the eigenvectors 
    eigen, s, v = np.linalg.svd(images_array.transpose(), full_matrices=False)
    #eigen = 1080000*415
    #s = 415*1
    #v = 415*415

    # plot eigenfacce
    for i in range(5):
        eigenface = process(-eigen[:,i]).reshape(600,600,3)
        io.imsave(str(i) + '_eigenface.jpg', eigenface, quality=100)

    ############ For prob 1.d ############
    for i in range(5):
        number = s[i] * 100 / sum(s)
        print(i, ' Eigenfaces: ',number, round(number, 1))
    return(eigen)

if __name__ == "__main__":
    
    #load data
    images_array = list()
    for n in range(415):
        path = os.path.join(images_path, str(n)+'.jpg')
        img = io.imread(path)
        images_array.append(img.flatten())
    images_array = np.array(images_array).astype('float32')
    
    average_face(images_array)
    eigen = eigen_face(images_array)
    reconstruction(images_array, eigen)