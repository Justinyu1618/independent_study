import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.manifold import TSNE
from datetime import datetime

import sys
sys.path.insert(0, './machine_learning_examples')
from rnn_class.brown import get_sentences_with_word2idx_limit_vocab, get_sentences_with_word2idx
from nlp_class2.util import find_analogies
from sklearn.feature_extraction.text import TfidfTransformer

print("updated")
def main():
	sentences, word2idx = get_sentences_with_word2idx_limit_vocab(n_vocab=1500)
	with open('w2v_word2idx.json','w') as f:
		json.dump(word2idx,f)
	
	V = len(word2idx)
	N = len(sentences)
	
	A = np.zeros((V,N))
	j = 0
	for sentence in sentences:
		for word in sentence:
			A[word,j] += 1
		j += 1
	print("finished word vectors")
	
	transformer = TfidfTransformer()
	A = transformer.fit_transform(A)
	
	A = A.toarray()
	
	idx2word = {v:k for k, v in word2idx.items()}
	
	tsne = TSNE()
	Z = tsne.fit_transform(A)
	"""
	plt.scatter(Z[:,0], Z[:,1])
	for i in range(V):
		#try:
		if True:
			plt.annotate(s=idx2word[i].encode("utf8"), xy=(Z[i,0], Z[i,1]))
		#except:
		#	print("bad string:", idx2word[i])
	plt.show()
	"""
	#tsne = TSNE(n_components=3)
	#We = tsne.fit_transform(A)
	We = Z
	print("starting analogies")
	print(find_analogies('king','man','woman', We, word2idx))
	print(find_analogies('france','paris','london', We, word2idx))
	print(find_analogies('france','paris','rome', We, word2idx))
	print(find_analogies('paris','france','italy', We, word2idx))
	
if __name__ == '__main__':
	main()
	
