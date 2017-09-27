import nltk
import os
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.decomposition import TruncatedSVD


wordnet_lemmatizer = WordNetLemmatizer()

data_address = '/home/justinyu/Desktop/Datasets/large-movie-reviews-dataset-master/acl-imdb-v1/train/'
urls = [line.rstrip() for line in open(data_address + 'urls_unsup.txt')]

all_reviews = {}

stopwords = set(stopwords.words('english'))

def my_tokenizer(s):
	s = s.lower()
	tokens = nltk.tokenize.word_tokenize(s)
	tokens = [wordnet_lemmatizer.lemmatize(t) for t in tokens]
	tokens = [t for t in tokens if t not in stopwords]	
	tokens = [t for t in tokens if not any(c.isdigit() for c in t)]
	tokens = [t for t in tokens if len(t) > 2 and '\'' not in t]
	return tokens
	

for x in range(100):
	if not urls[x] in all_reviews:
		all_reviews[urls[x]] = []
	all_reviews[urls[x]].append(my_tokenizer(open(data_address + 'unsup/' + str(x) + '_0.txt','r').read().replace('<br />','')))
	

all_reviews_tokens = {}
word_index_map = {}

for key in all_reviews.keys():
	all_reviews_tokens[key] = set([tokens for reviews in all_reviews[key] for tokens in reviews]) #HOLY SHIT WTF
	#word_index_map[key] = enumerate(all_reviews_tokens[key])
	
def token_to_vector(tokens, key):
	corpus = all_reviews_tokens[key]
	x = np.zeros((1,len(corpus)))
	for t in tokens:
		if t in corpus:
			x[0][list(corpus).index(t)] += 1
	return x

matrix = {}
for key in all_reviews_tokens:
	matrix[key] = np.zeros((len(all_reviews[key]),len(all_reviews_tokens[key])))
	for i in range(len(matrix[key])):
		matrix[key][i,:] = token_to_vector(all_reviews[key][i], key)

components = {}
svd = TruncatedSVD(n_components=10)
for x in matrix:
	svd.fit_transform(matrix[x])
	components[x] = svd.components_



for key in matrix:
	print("\n" + key)
	for i, comp in enumerate(components[key]):
		wordList = list(zip(all_reviews_tokens[key], comp))
		sortedList = sorted(wordList, key= lambda x: x[1], reverse=True)[:10]
		print("\nConcept: " + str(i))
		for x in sortedList:
			print(x)

		
"""	
for movie in all_reviews:
	i = 0
	for review in all_reviews[movie]:
		all_reviews[movie][i] = my_tokenizer(review)
		i += 1

"""

"""




word_index_map = {}
current_index = 0
all_tokens = []
all_titles = []
index_word_map = [] 

for title in titles:
	try:
		title = title.encode('ascii', 'ignore').decode()
		
		all_titles.append(title)
		tokens = my_tokenizer(title)
		all_tokens.append(tokens)
		for token in tokens:
			if token not in word_index_map:
				word_index_map[token] = current_index
				current_index += 1
				index_word_map.append(token)
	except:
		pass

	

	
N = len(all_tokens)
D = len(word_index_map)
X = np.zeros((N,D))
i = 0

for tokens in all_tokens:
	X[i,:] = token_to_vector(tokens)
	i += 1

svd = TruncatedSVD(n_components=10)
Z = svd.fit_transform(X)
print(svd.components_.shape)
print(len(svd.components_), len(svd.components_[0]))

for i, comp in enumerate(svd.components_):
	wordList = list(zip(index_word_map, comp))
	sortedList = sorted(wordList, key= lambda x: x[1], reverse=True)[:10]
	print("\nConcept: " + str(i))
	for x in sortedList:
		print(x)


#print(index_word_map)
"""

