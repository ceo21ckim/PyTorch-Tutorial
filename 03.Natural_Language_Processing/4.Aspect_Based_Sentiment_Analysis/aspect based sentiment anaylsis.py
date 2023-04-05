import pandas as pd 
import numpy as np 
import nltk 
import stanfordnlp

from nltk.corpus import stopwords, wordnet 
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem.wordnet import WordNetLemmatizer

MODELS_DIR = '.'
stanfordnlp.download('en', MODELS_DIR)
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

text = 'The Sound Quality is great but the battery life is very bad.'

txt = text.lower()
senList = nltk.sent_tokenize(txt)

for line in senList:
    txt_list = nltk.word_tokenize(line)
    taggedList = nltk.pos_tag(txt_list)
print(taggedList)


# we need to handle that first by joining multiple words features into a one-word feature.
# concatnate 'NN' and 'NN' 

newwordList = []
flag = 0 
for i in range(0, len(taggedList)-1):
    if (taggedList[i][1] == 'NN' and taggedList[i+1][1] == 'NN'):
        newwordList.append(taggedList[i][0] + taggedList[i+1][0])
        flag += 1

    else:
        if (flag==1):
            flag = 0 
            continue 
            
        newwordList.append(taggedList[i][0])

        if (i==len(taggedList)-2):
            newwordList.append(taggedList[i+1][0])

finaltxt = ' '.join(word for word in newwordList)

stop_word = set(stopwords.words('english'))
new_txt_list = nltk.word_tokenize(finaltxt)
wordsList = [word for word in new_txt_list if word not in stop_word]
taggedList = nltk.pos_tag(wordsList)

nlp = stanfordnlp.Pipeline(models_dir = MODELS_DIR)
doc = nlp(finaltxt)
doc