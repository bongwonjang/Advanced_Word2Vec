from subsampling import *
from huffman import *
from word2vec2 import *

import time
from collections import Counter

# argument presetting
ns = 3 # 0 for HS otherwise NS
mode = "CBOW"
 
corpus = load_data('text8', load_save=True, subsampling=False, sampling_rate=1e-8)
frequency = Counter(corpus)
processed = []

# Discard rare words
for word in corpus:
    if frequency[word] > 1:
        processed.append(word)

vocabulary = set(processed)

# Assign an index number to a word
word2ind = {}
i = 0
for word in vocabulary:
    word2ind[word] = i
    i += 1
ind2word = {}
for k, v in word2ind.items():
    ind2word[v] = k

print("Vocabulary size")
print(len(word2ind))
print("Corpus size")
print(len(processed))

################################################
# Code dict for hierarchical softmax
################################################
freqdict = {}
for word in vocabulary:
    freqdict[word] = frequency[word]

codedict = HuffmanCoding().build(freqdict)
nodedict = {}
ind2node = {}
i = 0
if ns == 0:
    for word in codedict[0].keys():
        code = codedict[0][word]
        s = ""
        nodeset = []
        codeset = []
        for ch in code:
            if s in nodedict.keys():
                nodeset.append(nodedict[s])
            else:
                nodedict[s] = i
                nodeset.append(i)
                i += 1
            codeset.append(int(ch))
            s += ch
        ind2node[word2ind[word]] = (nodeset, codeset)

start_time = time.time()

emb, _ = word2vec_trainer(ns, processed, word2ind, ind2word, freqdict, ind2node,
                          mode=mode, dimension=300, learning_rate=0.025, iteration= 100000) #0  * len(processed)

print("CBOW : --- %s min ---" % ((time.time() - start_time) / 60))

torch.save([emb, word2ind, ind2word], 'model.pt')

from analogical_task import *

emb, word2ind, ind2word = torch.load('model.pt')

analogical_task(word2ind, ind2word, emb)