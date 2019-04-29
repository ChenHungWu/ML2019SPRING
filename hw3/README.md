# This is hw3 of Machine Learning - Image Sentiment Classification

Task: Classify images to one of seven sentiment

Dataset: 28709 processed gray scale images with size 48*48
<br /><br />

cnn.py :use 15,033,863 params

dnn.py :for report only, the prediction rate is so....terrible.

confusion_matrix.py :modify line 7 8 9 to get confusion matrix, be sure to use **Y_train_valid_label** rather than **Y_train_valid**
<br /><br />
    

Usage of Scripts:

(1) Run hw3_train.sh:

    bash ./hw3_train.sh $1
    
    $1 : train.csv

(2) Run hw3_test.sh:

    bash ./hw3_test.sh $1 $2
    
    $1 : test.csv
    $2 : predict file (output file)

This script will download seven models to ensemble prediction.
