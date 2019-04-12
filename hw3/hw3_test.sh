
# $1 test.csv
# $2 predict file (output file)
wget https://github.com/WuChenHung/ML2019SPRING/releases/download/hw3/en01.h5
wget https://github.com/WuChenHung/ML2019SPRING/releases/download/hw3/en02.h5
wget https://github.com/WuChenHung/ML2019SPRING/releases/download/hw3/en03.h5
wget https://github.com/WuChenHung/ML2019SPRING/releases/download/hw3/en04.h5
wget https://github.com/WuChenHung/ML2019SPRING/releases/download/hw3/en05.h5
wget https://github.com/WuChenHung/ML2019SPRING/releases/download/hw3/en06.h5
wget https://github.com/WuChenHung/ML2019SPRING/releases/download/hw3/en07.h5
python3 testmodel.py $1 $2
