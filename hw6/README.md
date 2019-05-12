# HW6 of Machine Learning - Malicious Comments Identification
* Ranked 20 place in class

Task : use **word embedding** and **RNN** to identify whether the message in DCARD is malicious

Dataset : 120K message in DCARD

train_model.py : use word embedding and RNN

test_model.py : ensemble four models

bow_dnn.py : for report, use **bag of word** and **DNN** to train the model

prediction.py : prdiction your model 

<br><br>
    
Usage of Scripts:

(1) Run hw6_test.sh:

    bash ./hw6_test.sh $1 $2 $3

    $1 test_x.csv dir
    $2 dict.txt.big dir
    $3 output file dir

(2) Run hw6_train.sh:

    bash ./hw6_train.sh $1 $2 $3 $4
    
    $1 train_x.csv dir
    $2 train_y.csv dir
    $3 test_x.csv dir
    $4 dict.txt.big dir
