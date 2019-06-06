import sys 
from sklearn.externals import joblib
from keras.models import model_from_json
from keras.models import load_model
model_name = sys.argv[1]
json_file = open(model_name+'.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
model.load_weights(model_name+'.weight')
model.save(model_name+'.model')
model = load_model(model_name+'.model')
joblib.dump(model, 'compress_'+model_name+'.model', compress=9)