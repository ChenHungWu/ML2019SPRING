import sys
from PIL import Image
from torchvision import transforms
import numpy as np
import torch
import torch.nn as nn
from torchvision.models import resnet50, resnet101, vgg16, vgg19, densenet121, densenet169
#tar -zcvf ../images.tgz *.png
start = 0
pic_num = 200
img_address = sys.argv[1]
save_address = sys.argv[2]

model_number = 0
#[0:resnet50 1:resnet101 2:vgg16 3:vgg19 4:densenet121 5:densenet169]

device = torch.device("cuda" if (torch.cuda.is_available()) else "cpu")
mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])
data_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)])
data_untransform = transforms.Compose([ transforms.Normalize((-mean / std), (1.0 / std))])

if( model_number == 0 ):
    model = resnet50(pretrained=True)
elif( model_number == 1 ):
    model = resnet101(pretrained=True)
elif( model_number == 2 ):
    model = vgg16(pretrained=True)
elif( model_number == 3 ):
    model = vgg19(pretrained=True)
elif( model_number == 4 ):
    model = densenet121(pretrained=True)
elif( model_number == 5 ):
    model = densenet169(pretrained=True)

model.to(device)
model.eval();

# loss criterion
loss_func = nn.CrossEntropyLoss()

epsilon = 0.5/(255*0.224) #because normalized
count = 0 
for i in range(start, start+pic_num):
    img = Image.open(img_address+'/{:0>3d}.png'.format(i))#input
    img = data_transform(img).reshape((1, 3, 224, 224))

    # Send the data and label to the device
    img = img.to(device)
    
    # Set requires_grad attribute of tensor. Important for Attack
    img.requires_grad = True

    # Forward pass the data through the model
    before_pred = model(img)
    before_label = before_pred.max(1, keepdim=True)[1]# get the index of the max log-probability
    before_probability = nn.Softmax(dim=1)(before_pred)[0, before_label].item()

    total_delta = torch.zeros_like(img)

    epoch = 10
    L_infinity = 0.5/(255*0.224) #because normalized
    
    #for special case
    if i in [61,156] :
        L_infinity = 1/(255*0.224) 
    if i == 98:
        epoch = 80
    if i == 121:
        L_infinity = 2/(255*0.224) 
        epoch = 20
    
    for t in range(epoch):
        pred = model(img + total_delta)
        loss = loss_func(pred, before_label[0])
    
        # Zero all existing gradients
        model.zero_grad()

        # Calculate gradients of model in backward pass
        loss.backward()

        # Collect datagrad
        delta = img.grad.data

        # Collect the element-wise sign of the data gradient
        sign_delta = delta.sign()

        total_delta += epsilon * sign_delta

        total_delta = torch.clamp(input= total_delta, min= -L_infinity, max= L_infinity)

    attacked_image = img + total_delta

    # Re-classify the perturbed image
    pred = model(attacked_image)
    '''
    if(i==0):
        b_array = before_pred.cpu().detach().numpy()
        a_array = pred.cpu().detach().numpy()
        np.save('b_array.npy', b_array)
        np.save('a_array.npy', a_array)
    '''
    after_label = pred.max(1, keepdim=True)[1] # get the index of the max log-probability

    print("Pic number: ", i,"Before: ", before_label.item(), before_probability 
                           ," After: ",  after_label.item(),nn.Softmax(dim=1)(pred)[0, after_label].item() ) 

    if(before_label.item()!=after_label.item()):
        count+=1
    else:
        print('       QQ       attack failed')
    
    attacked_image = data_untransform(attacked_image.cpu().reshape((3, 224, 224)))
    attacked_image = torch.clamp(attacked_image, 0, 1)
    attacked_image = transforms.ToPILImage()(attacked_image)
    attacked_image.save(save_address+'/{:0>3d}.png'.format(i))
print("Attack rate: ",count/pic_num)