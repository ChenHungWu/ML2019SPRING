import sys
from keras.models import load_model
import numpy as np
import jieba 
from gensim.models.word2vec import Word2Vec
from keras.preprocessing.sequence import pad_sequences
from keras.models import model_from_json
from keras import backend as K 

#p test_model.py ./data/test_x.csv ./dict.txt.big final_result.csv
X_test_address = sys.argv[1]
dictionary_address = sys.argv[2]
output_adress = sys.argv[3]
w2v_model = Word2Vec.load("word2vec_250.model")

def predict_model(X_test, model_name):
    json_file = open(model_name+'.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    model.load_weights(model_name+'.weight')
    Y_preds = model.predict(X_test)
    print(Y_preds)
    K.clear_session()
    del model
    return Y_preds

def predict_model_hdf5(X_test, model_name):
    model = load_model(model_name+'.h5')
    Y_preds = model.predict(X_test)
    print(Y_preds)
    K.clear_session()
    del model
    return Y_preds

def process_text(text):
    cutWords = []
    for w in text :
        if( not_Bnumber(w) ):
            cutWords.append(w)
    return cutWords

def not_Bnumber(text):
    if len(text)>1 and (text[0]=='B' or 'b') and text[1].isdigit():
        return False
    else:
        return True
# Load dict from TA
jieba.load_userdict(dictionary_address) 

test_x = []
with open(X_test_address,'r',encoding = 'utf-8') as f :
    lines = f.readlines()
    for i,line in enumerate(lines) :   
        if i != 0 :        # ignore first line "id,comment"
            test_x.append(line.split(',',1)[1])

cutWords = []
# cut test x
for x in test_x :
    # Using accurate mode
    setList = jieba.cut(x,cut_all=False)
    cutWords.append(process_text(setList))

embedding_matrix = np.zeros(  (len(w2v_model.wv.vocab.items())+1, w2v_model.vector_size) )
word2idx = {}

vocab_list = [(word, w2v_model.wv[word]) for word, _ in w2v_model.wv.vocab.items()]
for i, vocab in enumerate(vocab_list):
    word, vec = vocab
    embedding_matrix[i + 1] = vec
    word2idx[word] = i + 1

def text_to_index(corpus):
    new_corpus = []
    for doc in corpus:
        new_doc = []
        for word in doc:
            try:
                new_doc.append(word2idx[word])
            except:
                new_doc.append(0)
        new_corpus.append(new_doc)
    return np.array(new_corpus)

PADDING_LENGTH = 100
X_test = text_to_index(cutWords)
X_test = pad_sequences(X_test, maxlen=PADDING_LENGTH, padding='post', truncating='post')

Y_preds0 = predict_model(X_test, 'rnn250_00')
Y_preds1 = predict_model(X_test, 'rnn250_01')
Y_preds2 = predict_model(X_test, 'rnn250_02')
Y_preds7 = predict_model(X_test, 'rnn250_07')
Y_preds_label = (Y_preds0 + Y_preds1 + Y_preds2 + Y_preds7)/4 

for i in range(Y_preds_label.shape[0]):
    if(Y_preds_label[i]>=0.5):
        Y_preds_label[i] = 1
    else:
        Y_preds_label[i] = 0

save = open(output_adress,'w')
save.write("id,label\n")
for i in range(len(Y_preds_label)):
    save.write(str(i) + "," + str(int(Y_preds_label[i])) + "\n")
save.close()

'''
START_TIME = time.time()
Y_preds0 = predict_model(X_test, 'rnn250_00')
runTime = time.time() - START_TIME
print(runTime)
np.save('rnn250_00.npy', Y_preds0)
'''