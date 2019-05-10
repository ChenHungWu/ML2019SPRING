# This is hw2 of Machine Learning - Winner or Loser

Dataset and Task Introduction

Task: Binary Classification Determine whether people can makes over 50K a year.

* R
Dataset: ADULT Extraction was done by Barry Becker from the 1994 Census database. A set of reasonably clean records was extracted using the following conditions: ((AGE>16) && (AGI>100) && (AFNLWGT>1) && (HRSWK>0)).

Reference: https://archive.ics.uci.edu/ml/datasets/Adult

Three methods:
1. logistic mode
2. generative model
3. best model (use sklearn.ensemble.GradientBoostingClassifier to training)

Usage of Scripts:

(1) Run logistic/generative:

    bash ./hw2_logistic.sh $1 $2 $3 $4 $5 $6 

    bash ./hw2_generative.sh $1 $2 $3 $4 $5 $6 

(2) Run best model:

    bash ./hw2_best.sh $1 $2 $3 $4 $5 $6 

Both scripts export only the prediction file. All the attribute are describe as follows:

+ $1: raw data (train.csv)  
+ $2: test data (test.csv)  
+ $3: provided train feature (X_train)  
+ $4: provided train label (Y_train)
+ $5: provided test feature (X_test)     
+ $6: prediction.csv
