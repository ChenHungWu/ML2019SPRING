import sys
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
from keras import backend as K

input_path = sys.argv[1]
output_path = sys.argv[2]
pic_size = 48
model_name = 'en05.h5'

def normalize(x):
    # utility function to normalize a tensor by its L2 norm
    return x / (K.sqrt(K.mean(K.square(x))) + 1e-7)

# read model
model = load_model(model_name)

layer_dict = dict([layer.name, layer] for layer in model.layers)
input_img = model.input

name_ls = ['leaky_re_lu_1']
#name_ls = ['leaky_re_lu_1', 'leaky_re_lu_2', 'leaky_re_lu_3', 'leaky_re_lu_4']
collect_layers = [ layer_dict[name].output for name in name_ls ]

nb_filter = 32
num_steps = 300
for cnt, c in enumerate(collect_layers):
    #print('Process layer {}'.format(collect_layers[cnt]))
    filter_imgs = []
    for filter_idx in range(nb_filter):
        input_img_data = np.random.random((1, 48, 48, 1)) 
        target = K.mean(c[:, :, :, filter_idx])
        grads = normalize(K.gradients(target, input_img)[0])
        iterate = K.function( [input_img, K.learning_phase()], [target, grads] )

        # calculate img
        input_image_data = np.copy(input_img_data)
        learning_rate = 0.05
        for i in range(num_steps):
            target, grads_val = iterate([input_image_data, 0])
            input_image_data += grads_val * learning_rate
        filter_imgs.append(input_image_data)

    #start plot
    fig , ax = plt.subplots( nrows= nb_filter//8, ncols = 8 , figsize=(14, 8))
    plt.tight_layout()
    for i in range(nb_filter) :
        ax[i//8,i%8].imshow( filter_imgs[i][0].reshape(48, 48, 1).squeeze(), cmap='BuGn' )
        ax[i//8,i%8].set_xticks([])
        ax[i//8,i%8].set_yticks([])
        ax[i//8,i%8].set_title('filter '+str(i)) 
    fig.savefig(output_path+'fig2_1.jpg')

