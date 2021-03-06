import nltk
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from nltk.stem import WordNetLemmatizer
from sklearn.decomposition import TruncatedSVD

wordnet_lemmatizer = WordNetLemmatizer()

titles = [line.rstrip() for line in open("all_book_titles.txt")]

stopwords = set(w.rstrip() for w in open('stopwords.txt'))
stopwords = stopwords.union({
	'introduction', 'edition', 'series', 'application',
    'approach', 'card', 'access', 'package', 'plus', 'etext',
    'brief', 'vol', 'fundamental', 'guide', 'essential', 'printed',
    'third', 'second', 'fourth'
})


def my_tokenizer(s):
	s = s.lower()
	tokens = nltk.tokenize.word_tokenize(s)
	tokens = [t for t in tokens if len(t) > 2]
	tokens = [wordnet_lemmatizer.lemmatize(t) for t in tokens]
	tokens = [t for t in tokens if t not in stopwords]
	tokens = [t for t in tokens if not any(c.isdigit() for c in t)]
	
	return tokens

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

	
def token_to_vector(tokens):
	x = np.zeros((1,len(word_index_map)))
	for t in tokens:
		i = word_index_map[t]
		x[0][i] = 1
	return x
	
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



plt.scatter(Z[:,0], Z[:,1])
for i in range(D):
	plt.annotate(index_word_map[i], xy=(Z[i,0], Z[i,1]))
plt.show()


