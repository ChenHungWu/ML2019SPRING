wget https://github.com/WuChenHung/ML2019SPRING/releases/download/hw8/label_probability.csv
python3 train_model.py $1 new_model
python3 compress_model.py new_model