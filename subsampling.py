import numpy as np
import math
import random
import json
from collections import Counter

########################################################################
# Subsampling.py
# 
# 처리 순서
# load_save = False
# 1. 코퍼스로부터 단어들을 토큰화
#    lower, replace, split 함수 사용
# 2. 단어 빈도 수에 따라 (단어 : 단어 빈도 수) 배열 생성
#    collections의 Counter()를 이용해 [('a', 5), ('b', 3) .. ]으로 변환
# 3. 단어 출현 확률을 이용해서 word2ind, ind2word, vocab를 구현
#    np.random에서 sample() 함수를 이용해 P(w_i) 생성
#    1 - np.sqrt(~~) 식과 비교하여 P(w_i)가 1 - np.sqrt(~~)보다 작으면
#    추가하지 않음
# 4. 전처리를 마친 corpus를 np 파일로 저장 후 list 형태로 리턴
#
# load_save = True
# 1. corpus를 저장했던 np 파일로부터 읽어들임
# 2. corpus를 list 형태로 리턴
########################################################################


def read_corpus(file_path):
    with open(file_path, 'r') as f:
        corpus = f.readline()

    return corpus

def subsampling_process(splited_corpus, sampling_rate=0.00001):
    frequency = Counter(splited_corpus)
    total = len(splited_corpus)

    processed = []
    for word in splited_corpus:
        ####################################
        #   f(w) = counter(w) / total
        #   t    = threshold = sampling_rate 
        #   P(w) = 1 - sqrt(t / f(w)) = 1 - sqrt(t * total / counter(w))
        ####################################
        p_w = 1.0 - math.sqrt(sampling_rate * total / frequency[word])
        sampling = random.random() # 0.0 ~ 1.0
        
        # print(p_w, sampling)

        if (sampling >= p_w):
            # 해당 단어는 살린다.
            processed.append(word)

    print('inital corpus len :', len(splited_corpus), ' ////  subsampled corpus len :', len(processed), )
    print('\nsubsampled rate :', len(processed) / len(splited_corpus))

    return processed

def normal_process(splited_corpus):
    frequency = Counter(splited_corpus)
    
    processed = []
    # Discard rare words
    # frequency가 1 이하인 word들은 전부 " "으로 처리
    for word in splited_corpus:
        if frequency[word] > 1:
            processed.append(word)
        else:
            processed.append(" ")

    return processed

########################################################################
# load_data : file_path로부터 데이터를 불러들인 후, 
#             Subsampling 전처리 과정을 한 훈련 데이터를 리턴함
# <input>
#  file_path    : 파일 경로. 1줄로된 텍스트만이 들어온다고 가정
#  load_save    : 이미 저장된 훈련 데이터를 리턴
#  subsampling  : Subsampling을 할 것인지 아닌지 결정
#
# <output>
#  corpus       : 단어를 ind로 매핑한 corpus
#
########################################################################
def load_data(file_path, load_save=False, subsampling=True, sampling_rate=0.00001):

    corpus = None
    word2ind = {}
    ind2word = {}
    suffix = '_subsampling' if subsampling else '_normal'

    if load_save:
        print('loading exising data....')
        corpus = np.load('preprocessed_corpus'+suffix+'.npy')

        print('finished loading data')

        # numpy로 읽었기 때문에, 명시적으로 list로 변환해서 보내준다.
        # 코드 이해를 위해서..
        return corpus.tolist()

    else:
        print('reading corpus.... from file:', file_path)
        corpus = read_corpus(file_path)
        
        print('word tokenizing....')
        corpus = corpus.lower() # 일괄 소문자
        corpus = corpus.replace('.', ' .') # "abc." -> "abc ."
        splited_corpus = corpus.split() # "ab cd" -> ["ab", "cd"]
        
        ####################################
        #   REMOVE THIS BEFORE EXECUTION   #
        ####################################
        # splited_corpus = splited_corpus[:100]

        if subsampling:
            preprocessed_corpus = subsampling_process(splited_corpus, sampling_rate=sampling_rate)
        else:
            preprocessed_corpus = normal_process(splited_corpus)

        print('saving existing data....')
        np.save('preprocessed_corpus'+suffix+'.npy', preprocessed_corpus)

        print('finished saving data')

        return preprocessed_corpus




