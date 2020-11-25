import codecs
import joblib
import numpy as np
import keras
import re
from unicodedata import normalize
import string
import pickle
from keras.utils.vis_utils import plot_model
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Embedding
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
from keras.callbacks import ModelCheckpoint

def max_len(lines):
	return max([len(line.split()) for line in lines])

def reshape_dataset(lang):
    n_list = []
    for sentence in lang:
        for word in sentence:
            n_str = ' '.join(word for word in sentence)
        n_list.append(n_str)
    return np.array(n_list)

def tokenize(lang):
    lang_tokenizer = keras.preprocessing.text.Tokenizer(filters='')
    lang_tokenizer.fit_on_texts(lang)
    return lang_tokenizer
        

def load_dataset(path,num_examples):
    lines = joblib.load(open(path, 'rb'))[:num_examples]
    inp_lang,targ_lang = zip(*lines)

    inp_lang = reshape_dataset(inp_lang)
    targ_lang = reshape_dataset(targ_lang)

    inp_lang_tokenizer = tokenize(inp_lang)
    targ_lang_tokenizer = tokenize(targ_lang)

    return inp_lang,targ_lang,inp_lang_tokenizer, targ_lang_tokenizer

def convert(lang, tensor):
  for t in tensor:
    if t!=0:
      print ("%d ----> %s" % (t, lang.index_word[t]))

def encode_input(tokenizer,lang):
    max_length = max_len(lang)
    tensor = tokenizer.texts_to_sequences(lang)
    tensor = keras.preprocessing.sequence.pad_sequences(tensor,padding='post',maxlen=max_length)
    return tensor

def encode_output(lang,vocab_size):
    ylist = []
    for sentence in lang:
        encoded = keras.utils.to_categorical(sentence,num_classes=vocab_size)
        ylist.append(encoded)
    y = np.array(ylist)
    y = y.reshape(lang.shape[0],lang.shape[1],vocab_size)
    return y

def define_model(inp_vocab,targ_vocab,inp_length,targ_length,n_units):
    model = Sequential()
    model.add(Embedding(inp_vocab, n_units, input_length=inp_length,mask_zero=True))
    model.add(LSTM(n_units))
    model.add(RepeatVector(targ_length))
    model.add(LSTM(n_units,return_sequences=True))
    model.add(TimeDistributed(Dense(targ_vocab,activation='softmax')))
    return model

inp_lang, targ_lang, inp_lang_tokenizer, targ_lang_tokenizer = load_dataset('tr-en-train.pkl',20000)

inp_lang_vocab = len(inp_lang_tokenizer.word_index) + 1
targ_lang_vocab = len(targ_lang_tokenizer.word_index) + 1
inp_lang_length = max_len(inp_lang)
targ_lang_length = max_len(targ_lang)

print('Turkish Vocabulary Size: %d' % inp_lang_vocab)
print('Turkish Max Length: %d' % (inp_lang_length))

print('English Vocabulary Size: %d' % targ_lang_vocab)
print('English Max Length: %d' % (targ_lang_length))


trainX = encode_input(inp_lang_tokenizer,inp_lang)
trainY = encode_input(targ_lang_tokenizer,targ_lang)
trainY = encode_output(trainY,targ_lang_vocab)

inp_lang_val, targ_lang_val, _, _ = load_dataset('tr-en-dev.pkl',None)

validX = encode_input(inp_lang_tokenizer,inp_lang)
validY = encode_input(targ_lang_tokenizer,targ_lang)
validY = encode_output(validY,targ_lang_vocab)

model = define_model(inp_lang_vocab,targ_lang_vocab,inp_lang_length,targ_lang_length,256)
model.compile(optimizer='adam', loss='categorical_crossentropy')
print(model.summary())
plot_model(model, to_file='model.png', show_shapes=True)

filename = 'model.h5'
checkpoint = ModelCheckpoint(filename, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
model.fit(trainX, trainY, epochs=30, batch_size=64, validation_data=(validX, validY), callbacks=[checkpoint], verbose=2)