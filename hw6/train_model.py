import sys
import numpy as np
from gensim.models.word2vec import Word2Vec
from keras.preprocessing.sequence import pad_sequences
import jieba 
from keras.models import load_model
from keras.models import Sequential
from keras import optimizers
from keras.layers import Embedding, Dense, Dropout, Activation, Flatten, LSTM, Bidirectional, LeakyReLU, GRU, BatchNormalization
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint, CSVLogger
#from sklearn.model_selection import train_test_split
#p train_model.py ./data/train_x.csv ./data/train_y.csv ./data/test_x.csv ./dict.txt.big
seed = 40666888
np.random.seed(seed)

X_train_address = sys.argv[1]
Y_train_address = sys.argv[2]
X_test_address = sys.argv[3]
path_dict = sys.argv[4]
save_model_name = 'rnn.h5'
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

# read test x
with open(X_test_address,'r',encoding = 'utf-8') as f :
    lines = f.readlines()
    for i,line in enumerate(lines) :   
        if i != 0 :        # ignore first line "id,comment"
            X_test.append(line.split(',',1)[1])

# read train label
with open(Y_train_address,'r',encoding = 'utf-8') as f :
    lines = f.readlines()
    for i,line in enumerate(lines) :
        if i != 0 :        # ignore first line "id,comment"
            Y_train.append( int((line.split(',',1)[1])[0]) ) 
Y_train = Y_train[0:119018]
#np.save('Y_train_250.npy', np.array(Y_train))

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
'''
# if cut the word without jeiba
cutWords = []
for i in range(len(X_train)):
    tt = []
    for j in range(len(X_train[i])):
        if(X_train[i][j] != '\n'):
            tt.append(X_train[i][j])
    cutWords.append(tt)
cutWords = cutWords[0:119018]
for i in range(len(X_test)):
    tt = []
    for j in range(len(X_test[i])):
        if(X_test[i][j] != '\n'):
            tt.append(X_test[i][j])
    cutWords.append(tt)
print(cutWords[0])
'''
#w2v_model = Word2Vec(cutWords, size=250, window=6, iter=50, min_count=3, workers=8)
#w2v_model.save("word2vec_250.model")

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
X_all = text_to_index(cutWords)
X_all = pad_sequences(X_all, maxlen=PADDING_LENGTH, padding='post', truncating='post')

X_train = X_all[:-20000]
X_test = X_all[-20000:]


embedding_layer = Embedding(input_dim=embedding_matrix.shape[0],
                            output_dim=embedding_matrix.shape[1],
                            weights=[embedding_matrix],
                            trainable=False, input_shape=(100,) )

#X_train = np.load(X_train_address)
#Y_train = np.load(Y_train_address)

#X_train, X_validation, Y_train, Y_validation = train_test_split(X_train, Y_train, test_size=0.2, random_state=seed)
X_validation = X_train[-10000:]
X_train = X_train[:-10000]
Y_validation = Y_train[-10000:]
Y_train = Y_train[:-10000]

drop_rate =0.4
model = Sequential()
model.add(embedding_layer)
model.add(Bidirectional(LSTM(256, activation='tanh', kernel_initializer='Orthogonal', recurrent_initializer='Orthogonal', return_sequences=True, unit_forget_bias=False, dropout=drop_rate, recurrent_dropout=drop_rate), merge_mode='sum'))
model.add(Bidirectional(LSTM(256, activation='tanh', kernel_initializer='Orthogonal', recurrent_initializer='Orthogonal', return_sequences=False, unit_forget_bias=False, dropout=drop_rate, recurrent_dropout=drop_rate), merge_mode='sum'))
model.add(Dense(256))
model.add(Dropout(drop_rate))
model.add(LeakyReLU(0.2))
model.add(Dense(256))
model.add(Dropout(drop_rate))
model.add(LeakyReLU(0.2))
model.add(Dense(1))
model.add(Activation('sigmoid'))
adam = optimizers.Adam(lr=0.002, clipvalue=1, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0)

model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

#clipvalue=0.5 clipnorm=1.
checkpoint = ModelCheckpoint(filepath=save_model_name, monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
lrate = ReduceLROnPlateau(monitor='val_acc', factor=0.2, patience=5, verbose=1, mode='auto', epsilon=0.0001, cooldown=0, min_lr=0.00000001)
earlystopping = EarlyStopping(monitor='val_acc', patience=10, verbose=1, mode='auto')
csvlogger = CSVLogger('log_'+save_model_name+'.csv', append=False)

model.fit(x=X_train, y=Y_train, batch_size=512, epochs=5000, validation_data=(X_validation, Y_validation), 
          callbacks=[checkpoint, lrate, earlystopping, csvlogger], shuffle=True)
