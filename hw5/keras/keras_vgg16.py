import numpy as np
import sys
from keras.applications import vgg16
import keras.backend as K
from PIL import Image
#https://github.com/soumyac1999/FGSM-Keras
#https://medium.com/@sci218mike/%E5%9C%96%E7%89%87%E9%A0%90%E8%99%95%E7%90%86%E4%BD%BF%E7%94%A8keras-applications-%E7%9A%84-preprocess-input-6ef0963a483e
pic_num = 20
dis = 6
img_address = sys.argv[1]
save_address = sys.argv[2]
start = int(sys.argv[3])

# Inverse of the preprocessing and plot the image
def plot_img(i, x):
    """
    x is a BGR image with shape (224, 224, 3) 
    """
    t = np.zeros_like(x)
    t[:,:,0] = x[:,:,2]
    t[:,:,1] = x[:,:,1]
    t[:,:,2] = x[:,:,0]  
    t = np.clip((t+[123.68, 116.779, 103.939]), 0, 255)
    #scipy.misc.toimage(t).save('attacked_images/{:0>3d}.png'.format(i))
    img = Image.fromarray(np.uint8(t))
    img.save(save_address+'/{:0>3d}.png'.format(i))


model = vgg16.VGG16(weights='imagenet')
x = np.zeros((pic_num,224,224,3))

for i in range(start,start+pic_num):
    #x = np.zeros((1,224,224,3))
    img = Image.open(img_address+'/{:0>3d}.png'.format(i))#input
    x[i%pic_num] = np.array(img)

# Create a batch and preprocess the image
#x = image.img_to_array(img)
x = vgg16.preprocess_input(x)

# Get the initial predictions
pre_preds = model.predict(x)
initial_class = np.argmax(pre_preds,axis=1)

# Get current session (assuming tf backend)
sess = K.get_session()
# Initialize adversarial example with input image
x_adv = x

# Set variables
epochs = 10
epsilon = 1


target = K.one_hot(initial_class, 1000)
for j in range(epochs): 
    # One hot encode the initial class
    
    # Get the loss and gradient of the loss wrt the inputs
    loss = K.categorical_crossentropy(target, model.output)
    grads = K.gradients(loss, model.input)

    # Get the sign of the gradient
    delta = K.sign(grads[0])

    # Perturb the image
    x_adv = x_adv + epsilon*delta

    # Get the new image and predictions
    x_adv = sess.run(x_adv, feed_dict={model.input:x})

    diff = x - x_adv
    temp = np.clip(temp ,-dis, dis)
    x_adv = x - temp

    print('start :',start,' epochs :',j)

preds = model.predict(x_adv)
for i in range(pic_num):
	print(vgg16.decode_predictions(pre_preds, top=1)[i], vgg16.decode_predictions(preds, top=1)[i])

for i in range(start,start+pic_num):
	plot_img(i, x_adv[(i%pic_num)])

'''
Vgg-16 prediction attack rate: 0.925
Vgg-19 prediction attack rate: 0.82
ResNet-50 prediction attack rate: 0.505
ResNet-101 prediction attack rate: 0.46
DenseNet-121 prediction attack rate: 0.55
Densenet-169 prediction attack rate: 0.47
'''
