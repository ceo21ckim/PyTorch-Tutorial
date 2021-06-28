import nltk
nltk.download('book', quiet= True) # example
from nltk.book import *
from nltk import word_tokenize, pos_tag, ne_chunk, sent_tokenize
from nltk.corpus import gutenberg
from nltk import regexp_tokenize



# data_set
emma_raw = gutenberg.raw('austen-emma.txt')

text = emma_raw[:1000]


# word_tokenize = 토큰화
sentence1 = word_tokenize(text) # 단어를 split
sentence2 = sent_tokenize(text) # 문장을 splitS

len(sentence1)

len(sentence2)

# 문장으로 나눈 것을 단어로 나누고 길이를 합쳐도 동일한 결과값이 출력된다.
word_token = [word_tokenize(sentence) for sentence in sentence2]
length = 0 
for word in word_token:
    print(' 단어 개수 : {}'.format(len(word)))
    length += len(word)

print(length == len(sentence1))



# 불용어 제거 stop word elimination / is, the, a ,...
nltk.download('stopwords')

stopwords = nltk.corpus.stopwords.words('english')

filter_words = []
for word in sentence1:
    word = word.lower()
    if word not in stopwords:
        filter_words.append(word)
filter_words
# 89개의 stopwords가 filter 되었음을 알 수 있다.
len(sentence1) # 198
len(filter_words) # 109

# 어간 추출
# Stemming : 일반적인 방법이나 단순화된 방법을 적용하여 원래 단어에서 일부 철자가 훼손된 어근을 추출하는 경향이 존재
# Lemmatization : 문법적인 요소와 의미적인 부분을 감안해 정확한 철자로 된 어근을 찾아주지만, Stemming에 비해 긴 실행 시간이 필요


# stemming 의 경우 원형을 찾아내지 못하는 경우가 많다.
from nltk.stem import PorterStemmer, LancasterStemmer
poster = PorterStemmer()
lancaster = LancasterStemmer()


[poster.stem(w) for w in filter_words]
[lancaster.stem(w) for w in filter_words]


from nltk.stem import WordNetLemmatizer
wordnet = WordNetLemmatizer()

filter_lemma = [wordnet.lemmatize(w) for w in filter_words]

w = nltk.Text(filter_lemma)

import matplotlib.pyplot as plt


w.plot()

