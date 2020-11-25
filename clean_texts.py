import codecs
import re
import string
import numpy as np
import joblib
from unicodedata import normalize

def open_file(filename):
    with codecs.open(filename,encoding='utf-8') as f:
        text = f.read()
    return text.strip().split('\n')

## this method will combine two line by line
## e.g. list1[0] = "merhaba", list2[0] = "hi"
## will become new_list[0] = "merhaba hi" 
def combine_files(list1,list2,brace='\t'):
    new_list = []
    for line1,line2 in zip(list1,list2):
        new_list.append('%s%s%s'%(line1,brace,line2))
    return new_list 

## split into sentences
def to_pairs(list):
    return [line.split('\t') for line in list]

def clean_pairs(lines):
    cleaned = list()
    re_print = re.compile('[^%s]'% re.escape(string.printable))
    table = str.maketrans('','',string.punctuation)
    for pair in lines:
        clean_pair = []
        for line in pair:
            line = normalize('NFD',line).encode('ascii','ignore')
            line = line.decode('utf-8')
            line = line.split()
            line = [word.lower().translate(table) for word in line]
            line = [re_print.sub('',word) for word in line]
            line = [word for word in line if word.isalpha()]
            clean_pair.append(line)
        cleaned.append(clean_pair)
    return np.array(cleaned)

def save_data(sentences,filename):
    joblib.dump(sentences,open(filename,'wb'))
    print('Saved: %d' % len(sentences))

filename1 = 'opus-100-corpus-en-tr\opus.en-tr-test.tr'
filename2 = 'opus-100-corpus-en-tr\opus.en-tr-test.en'

file1 = open_file(filename1)
file2 = open_file(filename2)

combined = combine_files(file1,file2)

pairs = to_pairs(combined)

clean_pairs = clean_pairs(pairs)

filename = 'tr-en-test.pkl'
save_data(clean_pairs,filename)

for i in range(20):
	print('[%s] => [%s]' % (clean_pairs[i,0], clean_pairs[i,1]))