import sys
import numpy as np
from gensim.models.word2vec import Word2Vec
from keras.preprocessing.sequence import pad_sequences
import jieba 
from keras.utils import to_categorical
from keras.models import Sequential
from keras import optimizers
from keras.layers import Embedding, Dense, Dropout, Activation, Flatten, LSTM, Bidirectional, LeakyReLU, GRU, BatchNormalization
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint, CSVLogger
#from sklearn.model_selection import train_test_split
#p bow_test.py ./data/train_x.csv ./data/train_y.csv ./data/test_x.csv ./dict.txt.big
seed = 6
np.random.seed(seed)

X_train_address = sys.argv[1]
Y_train_address = sys.argv[2]
X_test_address = sys.argv[3]
path_dict = sys.argv[4]
save_model_name = 'bow_rnn.h5'
w2v_model = Word2Vec.load("word2vec_250.model")

####################################
X_train = []
X_test = []
Y_train = []

jieba.load_userdict(path_dict)

def not_Bnumber(text):
    if len(text)>1 and (text[0]=='B' or 'b') and text[1].isdigit():
        return False
    else:
        return True

def process_text(text):
    cutWords = []
    for w in text :
        if( not_Bnumber(w) ):            
            cutWords.append(w)
    return cutWords 

# read train x
with open(X_train_address,'r',encoding = 'utf-8') as f :
    lines = f.readlines()
    for i,line in enumerate(lines) :   
        if i != 0 :        # ignore first line "id,comment"
            X_train.append(line.split(',',1)[1])

# read train label
with open(Y_train_address,'r',encoding = 'utf-8') as f :
    lines = f.readlines()
    for i,line in enumerate(lines) :
        if i != 0 :        # ignore first line "id,comment"
            Y_train.append( int((line.split(',',1)[1])[0]) ) 
Y_train = Y_train[0:119018]
#np.save('Y_train_250.npy', np.array(Y_train))

# read test x
with open(X_test_address,'r',encoding = 'utf-8') as f :
    lines = f.readlines()
    for i,line in enumerate(lines) :   
        if i != 0 :        # ignore first line "id,comment"
            X_test.append(line.split(',',1)[1])

cutWords = []
# cut train x 
for x in X_train :
    # Using accurate mode
    setList = jieba.cut(x, cut_all=False)
    cutWords.append(process_text(setList))
cutWords = cutWords[0:119018]

# cut test x
for x in X_test :
    # Using accurate mode
    setList = jieba.cut(x,cut_all=False)
    cutWords.append(process_text(setList))
####################################################
#w2v_model = Word2Vec(cutWords, size=100, window=6, iter=20, min_count=5, workers=8)
#w2v_model.save("word2vec_100.model")
 
embedding_matrix = np.zeros(  (len(w2v_model.wv.vocab.items())+1, w2v_model.vector_size) )
word2idx = {}
vocab_list = []
for word , others in w2v_model.wv.vocab.items() :
    vocab_list.append((word , w2v_model.wv[word]))

for i , vocab in enumerate(vocab_list) :
    word , vector = vocab
    embedding_matrix[i+1] = vector
    word2idx[word] = i+1

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
X_all = text_to_index(cutWords)

train_x_wv_bow = np.zeros( (len(X_all), len(word2idx)+1), dtype=np.int8)
for i, sentence in enumerate(X_all) :
    for w in sentence :
        try :
            train_x_wv_bow[i,w] += 1 
        except :
            pass

X_train = train_x_wv_bow[:-20000]
X_test = train_x_wv_bow[-20000:]
#np.save('bow_test.npy',X_test)
Y_train = to_categorical(Y_train,2)

model = Sequential()
model.add(Dense(2048, input_shape=X_train[0].shape, activation='sigmoid'))
model.add(Dense(2048, activation='sigmoid'))
model.add(Dense(1024, activation='sigmoid'))
model.add(Dense(1024, activation='sigmoid'))
model.add(Dense(2, activation='softmax'))
adam = optimizers.Adam(lr=0.002, clipvalue=1, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0)
model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

#clipvalue=0.5 clipnorm=1.
checkpoint = ModelCheckpoint(filepath=save_model_name, monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
lrate = ReduceLROnPlateau(monitor='val_acc', factor=0.2, patience=5, verbose=1, mode='auto', epsilon=0.0001, cooldown=0, min_lr=0.00000001)
earlystopping = EarlyStopping(monitor='val_acc', patience=10, verbose=1, mode='auto')
csvlogger = CSVLogger('log_'+save_model_name+'.csv', append=False)

history = model.fit(x=X_train, y=Y_train, batch_size=256
    , epochs=5000, validation_split=0.15, shuffle=True, callbacks=[lrate, checkpoint, earlystopping, csvlogger])



