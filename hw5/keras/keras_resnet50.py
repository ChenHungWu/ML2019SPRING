import numpy as np
import sys
from keras.applications import resnet50
import keras.backend as K
from PIL import Image

pic_num = 20
dis = 6
img_address = sys.argv[1]
save_address = sys.argv[2]
start = int(sys.argv[3])

# Inverse of the preprocessing and plot the image
def plot_img(i, x):
    t = np.zeros_like(x)
    t[:,:,0] = x[:,:,2]
    t[:,:,1] = x[:,:,1]
    t[:,:,2] = x[:,:,0]  
    t = np.clip((t+[123.68, 116.779, 103.939]), 0, 255)
    img = Image.fromarray(np.uint8(t))
    img.save(save_address+'/{:0>3d}.png'.format(i))


model = resnet50.ResNet50(weights='imagenet')
x = np.zeros((pic_num,224,224,3))

for i in range(start,start+pic_num):
    img = Image.open(img_address+'/{:0>3d}.png'.format(i))#input
    x[i%pic_num] = np.array(img)

# Create a batch and preprocess the image
x = resnet50.preprocess_input(x)

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
    print(resnet50.decode_predictions(pre_preds, top=1)[i], resnet50.decode_predictions(preds, top=1)[i])

for i in range(start,start+pic_num):
    plot_img(i, x_adv[(i%pic_num)])

'''
Vgg-16 prediction attack rate: 0.4
Vgg-19 prediction attack rate: 0.4
ResNet-50 prediction attack rate: 0.91
ResNet-101 prediction attack rate: 0.505
DenseNet-121 prediction attack rate: 0.45
Densenet-169 prediction attack rate: 0.4
'''