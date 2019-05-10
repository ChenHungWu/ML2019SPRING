# $1 train_x.csv dir
# $2 train_y.csv dir
# $3 test_x.csv dir
# $4 dict.txt.big dir
wget https://github.com/WuChenHung/ML2019SPRING/releases/download/hw6/word2vec_250.model
wget https://github.com/WuChenHung/ML2019SPRING/releases/download/hw6/word2vec_250.model.trainables.syn1neg.npy
wget https://github.com/WuChenHung/ML2019SPRING/releases/download/hw6/word2vec_250.model.wv.vectors.npy
python3 train_model.py $1 $2 $3 $4