# This is hw5  of Machine Learning - Adversarial Attack

Dataset and Task Introduction

Task: untarget FGSM attack images to mislead trained model

Dataset: 200 RGB images with size 224*224

** only when you meet the model in black block, you can attack it easily ~
</br></br>
12 methods:
1. Using Pytorch 
    * vgg16, vgg19, resnet50, resnet101, densenet121, densenet169
    \ one can modify line 14 in pytorch_fgsm_attacked.py to change model
    
    Usage:
    ~~~~
    python ./pytorch_fgsm_attacked.py $1 $2
    
    $1 input img dir
    $2 output img dir
    ~~~~
    
2. Using keras
    * vgg16, vgg19, resnet50, resnet101, densenet121, densenet169
    6 different code in ./keras_attack
    Usage:
    ~~~~
    python ./keras_vgg16.py $1 $2
    
    $1 input img dir
    $2 output img dir
    ~~~~
    
Usage of Scripts:

(1) Run hw5_fgsm.sh:

    bash ./hw5_fgsm.sh $1 $2

    $1 input img dir
    $2 output img dir

(2) Run hw5_best.sh:

    bash ./hw5_best.sh $1 $2
    
    $1 input img dir
    $2 output img dir
