import numpy as np
import sys
from keras.applications import densenet
import keras.backend as K
from PIL import Image

pic_num = 20
dis = 5
img_address = sys.argv[1]
save_address = sys.argv[2]
start = int(sys.argv[3])

# Inverse of the preprocessing and plot the image
def plot_img(i, x):
    x *= [0.229, 0.224, 0.225]
    x += [0.485, 0.456, 0.406]
    x *= 255
    x = np.clip(x,0,255)
    img = Image.fromarray(np.uint8(x))
    img.save(save_address+'/{:0>3d}.png'.format(i))


model = densenet.DenseNet169(weights='imagenet')
x = np.zeros((pic_num,224,224,3))

for i in range(start,start+pic_num):
    img = Image.open(img_address+'/{:0>3d}.png'.format(i))#input
    x[i%pic_num] = np.array(img)

# Create a batch and preprocess the image
x = densenet.preprocess_input(x)

# Get the initial predictions
pre_preds = model.predict(x)
initial_class = np.argmax(pre_preds,axis=1)

# Get current session (assuming tf backend)
sess = K.get_session()
# Initialize adversarial example with input image
x_adv = x

# Set variables
epochs = 10
epsilon = 1/(255*0.225)

target = K.one_hot(initial_class, 1000)


#set constrain
t1 = np.array([dis,dis,dis])
t1 = t1/255
t1 /= [0.229, 0.224, 0.225]

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
    temp = np.zeros((pic_num,224,224,3))

    temp[:,:,:,0] = np.clip(diff[:,:,:,0],-t1[0],t1[0])
    temp[:,:,:,1] = np.clip(diff[:,:,:,1],-t1[1],t1[1])
    temp[:,:,:,2] = np.clip(diff[:,:,:,2],-t1[2],t1[2])
    x_adv = x - temp

    print('start :',start,' epochs :',j)
preds = model.predict(x_adv)
for i in range(pic_num):
    print(densenet.decode_predictions(pre_preds, top=1)[i], densenet.decode_predictions(preds, top=1)[i])

for i in range(start,start+pic_num):
    plot_img(i, x_adv[(i%pic_num)])

'''
Vgg-16 prediction attack rate: 0.35
Vgg-19 prediction attack rate: 0.295
ResNet-50 prediction attack rate: 0.34
ResNet-101 prediction attack rate: 0.335
DenseNet-121 prediction attack rate: 0.46
Densenet-169 prediction attack rate: 0.875
'''