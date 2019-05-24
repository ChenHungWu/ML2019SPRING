# $1 <images path> images/
# $2 <test_case.csv path> test_case.csv
# $3 <prediction file path> ans.csv
wget https://github.com/WuChenHung/ML2019SPRING/releases/download/hw7/encoder_model.h5
python3 cluster.py $1 $2 $3