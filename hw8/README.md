# HW8 of Machine Learning - Model Compression for Image Sentiment Classification

* It is a multi-class problem
* Ranked 2 place in class

Task: Use **MobileNet** to classify images to one of seven sentiment

Dataset: 28709 processed gray scale images with size 48*48
</br></br>
train_model.py : use 62955 params

compress_model.py : compress the model

test_model.py : get prediction
</br></br>

Usage of Scripts:

(1) Run hw8_train.sh:

    bash ./hw8_train.sh $1
    
    $1 <training data> data/train.csv

(2) Run hw8_test.sh:

    bash ./hw8_test.sh $1 $2
       
    $1 <testing data> data/test.csv
    $2 <prediction file> result.csv
    $3 <model_name> compress_model5.model
