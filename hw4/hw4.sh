
# $1 train.csv
# $2 output file
wget https://github.com/WuChenHung/ML2019SPRING/releases/download/hw3/en05.h5
python3 saliency_map.py $1 $2
python3 draw_lime.py $1 $2
python3 visualize_by_noise.py $1 $2
python3 visualize_by_picture.py $1 $2
