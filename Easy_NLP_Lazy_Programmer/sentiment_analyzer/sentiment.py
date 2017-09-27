import nltk
import numpy as np

from nltk.stem import WordNetLemmatizer
from sklearn.linear_model import LogisticRegression
from bs4 import BeautifulSoup

word_lemmatizer = WordNetLemmatizer()

stopwords = set(words.rstrip() for words in open("stopwords.txt"))

positive_reviews = BeautifulSoup(open('positive.review').read(), 'lxml')
positive_reviews = positive_reviews.findAll('review_text')

negative_reviews = BeautifulSoup(open('negative.review').read(), 'lxml')
negative_reviews = negative_reviews.findAll('review_text')

np.random.shuffle(positive_reviews)
positive_reviews = positive_reviews[:len(negative_reviews)]

word_index_map = {}
index = 0
pos_tokenized, neg_tokenized = [], []

def my_tokenizer(s):
	s = s.lower()
	tokens = nltk.tokenize.word_tokenize(s)
	tokens = [t for t in tokens if len(t) > 2]
	tokens = [word_lemmatizer.lemmatize(t) for t in tokens]
	tokens = [t for t in tokens if t not in stopwords]
	return tokens

for review in positive_reviews:
	tokens = my_tokenizer(review.text)
	pos_tokenized.append(tokens)
	for token in tokens:
		if token not in word_index_map:
			word_index_map[token] = index
			index += 1
			
for review in negative_reviews:
	tokens = my_tokenizer(review.text)
	neg_tokenized.append(tokens)
	for token in tokens:
		if token not in word_index_map:
			word_index_map[token] = index
			index += 1
			
def token_to_vector(tokens, label):
	x = np.zeros(len(word_index_map) + 1)
	for t in tokens:
		i = word_index_map[t]
		x[i] += 1
	x = x / x.sum()
	x[-1] = label
	return x
	
N = len(pos_tokenized) + len(neg_tokenized)

data = np.zeros((N, len(word_index_map)+1))
i = 0

for tokens in pos_tokenized:
	xy = token_to_vector(tokens, 1)
	data[i,:] = xy
	i += 1

for tokens in neg_tokenized:
	xy = token_to_vector(tokens, 0)
	data[i,:] = xy
	i += 1

np.random.shuffle(data)

X = data[:,:-1]
Y = data[:, -1]

Xtrain = X[:-100,]
Ytrain = Y[:-100,]
Xtest = X[-100:,]
Ytest = Y[-100:,]

model = LogisticRegression()
model.fit(Xtrain, Ytrain)
print("Classification rate:", model.score(Xtest, Ytest))

threshold = 0.5

"""
print(model.coef_)
for word, index in word_index_map.items():
	weight = model.coef_[0][index]
	if abs(weight) > threshold:
		print(word,weight)
"""

