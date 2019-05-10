# $1 test_x.csv dir
# $2 dict.txt.big dir
# $3 output file dir
wget https://github.com/WuChenHung/ML2019SPRING/releases/download/hw6/word2vec_250.model
wget https://github.com/WuChenHung/ML2019SPRING/releases/download/hw6/word2vec_250.model.trainables.syn1neg.npy
wget https://github.com/WuChenHung/ML2019SPRING/releases/download/hw6/word2vec_250.model.wv.vectors.npy
wget https://github.com/WuChenHung/ML2019SPRING/releases/download/hw6/rnn250_00.json
wget https://github.com/WuChenHung/ML2019SPRING/releases/download/hw6/rnn250_00.weight
wget https://github.com/WuChenHung/ML2019SPRING/releases/download/hw6/rnn250_01.json
wget https://github.com/WuChenHung/ML2019SPRING/releases/download/hw6/rnn250_01.weight
wget https://github.com/WuChenHung/ML2019SPRING/releases/download/hw6/rnn250_04.json
wget https://github.com/WuChenHung/ML2019SPRING/releases/download/hw6/rnn250_04.weight
wget https://github.com/WuChenHung/ML2019SPRING/releases/download/hw6/rnn250_07.json
wget https://github.com/WuChenHung/ML2019SPRING/releases/download/hw6/rnn250_07.weight
python3 test_model.py $1 $2 $3