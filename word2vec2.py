import torch
import numpy as np
import random
import argparse
from random import shuffle
from collections import Counter
from huffman import HuffmanCoding


def getRandomContext(corpus, C=5):
    wordID = random.randint(0, len(corpus) - 1)
    
    context = corpus[max(0, wordID - C):wordID]
    if wordID + 1 < len(corpus):
        context += corpus[wordID + 1:min(len(corpus), wordID + C + 1)]

    centerword = corpus[wordID]
    context = [w for w in context if w != centerword]

    if len(context) > 0:
        return centerword, context
    else:
        return getRandomContext(corpus, C)


def CBOW_HS(center, context, codes, inputMatrix, outputMatrix):

    v_wi = torch.sum(inputMatrix[context], dim=0)
    
    P_wi = torch.tensor([1.0])
    for i, code in enumerate(codes):
        lxl = 1 if code == 0 else -1
        P_wi = P_wi * torch.sigmoid(lxl * torch.dot(outputMatrix[i], v_wi))
    
    loss = -torch.log(P_wi)

    grad_out = torch.zeros(outputMatrix.size())
    grad_emb = torch.zeros(v_wi.size())
    for i, code in enumerate(codes):
        if code == 0:
            grad = torch.sigmoid(torch.dot(outputMatrix[i], v_wi)) - 1
            grad_out[i] = grad * v_wi
            grad_emb = grad_emb + grad * outputMatrix[i]
        else:
            grad = torch.sigmoid(torch.dot(outputMatrix[i], v_wi))
            grad_out[i] = grad * v_wi
            grad_emb = grad_emb + grad * outputMatrix[i]
            
            
    return loss, grad_emb, grad_out


def CBOW_NS(center, context, inputMatrix, outputMatrix, negative_samples):
    v_wi = torch.sum(inputMatrix[context], dim=0)  # [D]

    # 1. 이걸 수정. zeros를 해야, 나중에 W_out = W_out - lr * grad_out에서
    # center이랑, neg 이외 vector 부분이 업데이트 되는 것을 피할 수 있다.
    grad_out = torch.zeros(outputMatrix.size()) # [V, D]
    
    grad_emb = torch.zeros(v_wi.size()) # [D]

    # 2. P_wi를 구하는 방식을 수정
    # 원인으로 예상 : 곱셈을 먼저 진행한 다음에, 마지막에 log를 붙일 때,
    #                 소수값들을 계속 곱한 것이 되므로.. 0으로 소멸할 위험 가능..
    #                 
    # 수정한 방향   : 처음부터 -log(center) - log(sample1) .... 형태로 동작
    #                 전부다 -를 붙여야 한다. 가끔 +를 붙인데가 있는데..
    #                 그러면.. cross entropy loss가 아니다..

    # 긍정적 예 순전파 (context to center)
    P_wi = -torch.log(torch.sigmoid(torch.dot(outputMatrix[center], v_wi)))

    # 부정적 예 순전파 (context to center)
    for sample in negative_samples: # 네거티브 샘플들에 대한 순전파 수행
        P_wi -= torch.log(torch.sigmoid(-1 * torch.dot(outputMatrix[sample], v_wi)))
  
    loss = P_wi # 최종적으로 -log(center) + log(sample1) + log(sample2) ... 형태

    # 3. gradient를 구하는 것을, 명시적으로 하기 위해 순서대로 나눔.
    # W_out를 먼저 구하고, W_emb를 구하는 것이 눈으로 보기 편하다.
    
    # 4. W_out
    # cross entropy loss이므로, target에 대해서만 y - 1을 해서 grad를 구한다.
    # 그리고, 당연하지만, h * g^t를 해야 하는데... 여기서는 그냥 1:1 식으로 업데이트
    # grad_out은 초기 형태가 전부 0인 tensor이므로, 그냥 대입한다.
    grad = torch.sigmoid(torch.dot(outputMatrix[center], v_wi))
    grad_out[center] = (grad - 1) * v_wi

    # 그리고 neg에 대해서도 업데이트 해야 한다. 단순하지만, y - 1을 하지 "않은" 형태로
    # 곧이 곧대로, h * g^t를 해야 하는데.. 여기서도 1:1 식으로 업데이트한다.
    for sample in negative_samples:
        grad = torch.sigmoid(torch.dot(outputMatrix[sample], v_wi))
        grad_out[sample] = grad * v_wi

    # 5. W_emb
    # dE/dh이다. 왜냐하면, 어차피 dh/dWa는 1이기 때문에, 1은 곱셈에서 신경쓰지 않아도 된다.
    # 숙제 4의 backpropagation 항목 참조!
    # 중요한 것은..
    # dE/dh = (dE/dvh) * (dvh/dh)이다.
    # 이거를 풀어쓰면
    # dE/dh = Σ{ (grad - α) * v }
    # 이렇게 된다. α는.. grad - 1 또는 grad - 0에서 그 1과 0이다.
    # 앞 stage에서 전달되는 error에 있는 값이다.
    # 당연히 v가 center이면, 1이고, neg이면 0이다.
    grad = torch.sigmoid(torch.dot(outputMatrix[center], v_wi))
    grad_emb += (grad - 1) * outputMatrix[center]
    for sample in negative_samples:
        grad = torch.sigmoid(torch.dot(outputMatrix[sample], v_wi))
        grad_emb = grad * outputMatrix[sample]

    return loss, grad_emb, grad_out


def Skipgram_HS(center, context, codes, inputMatrix, outputMatrix):

    v_wi = inputMatrix[center]
    
    P_wi = torch.tensor([1.0])
    for i, code in enumerate(codes):
        lxl = 1 if code == 0 else -1
        P_wi = P_wi * torch.sigmoid(lxl * torch.dot(outputMatrix[i], v_wi))
    
    loss = -torch.log(P_wi)

    grad_out = torch.zeros(outputMatrix.size())
    grad_emb = torch.zeros(v_wi.size())
    for i, code in enumerate(codes):
        if code == 0:
            grad = torch.sigmoid(torch.dot(outputMatrix[i], v_wi)) - 1
            grad_out[i] = grad * v_wi
            grad_emb = grad_emb + grad * outputMatrix[i]
        else:
            grad = torch.sigmoid(torch.dot(outputMatrix[i], v_wi))
            grad_out[i] = grad * v_wi
            grad_emb = grad_emb + grad * outputMatrix[i]

    return loss, grad_emb, grad_out


def Skipgram_NS(center, context, inputMatrix, outputMatrix, negative_samples):

    v_wi = inputMatrix[center]  # [300]
    grad_out = torch.zeros(outputMatrix.size())  # [38076, 300]
    grad_emb = torch.zeros(v_wi.size())  # [300]

    # 긍정적 예 순전파 (center to context)
    P_wi = -torch.log(torch.sigmoid(torch.dot(v_wi, outputMatrix[context])))
    
    # 부정적 예 순전파 (center to context)
    for sample in negative_samples:  # 네거티브 샘플들에 대한 순전파 수행
        P_wi -= torch.log(torch.sigmoid(-1 * torch.dot(v_wi, outputMatrix[sample])))

    loss = P_wi

    # W_out
    grad = torch.sigmoid(torch.dot(outputMatrix[context], v_wi))
    grad_out[context] = (grad - 1) * v_wi
    for sample in negative_samples:
        grad = torch.sigmoid(torch.dot(outputMatrix[sample], v_wi))
        grad_out[sample] = grad * v_wi

    # W_emb
    grad = torch.sigmoid(torch.dot(outputMatrix[context], v_wi))
    grad_emb += (grad - 1) * outputMatrix[context]
    for sample in negative_samples:
        grad = torch.sigmoid(torch.dot(outputMatrix[sample], v_wi))
        grad_emb = grad * outputMatrix[sample]

    return loss, grad_emb, grad_out


def word2vec_trainer(ns, corpus, word2ind, ind2word, freqdict, ind2node,
                     mode="CBOW", dimension=64, learning_rate=0.05, iteration=50000):
    # initialization
    W_emb = torch.randn(len(word2ind), dimension) / (dimension ** 0.5)
    W_out = torch.randn(len(word2ind), dimension) / (dimension ** 0.5)
    W_emb = W_emb
    W_out = W_out
    window_size = 5

    ns_power = None
    words = None
    wordlist = None 
    freqlist = None 
    problist = None
    # Negative Sampling
    if ns != 0:    
        ns_power = 0.75
        words = [w for w in freqdict.keys()]
        wordlist = [word2ind[w] for w in words]
        freqlist = [freqdict[w] ** ns_power for w in words]
        problist = [f / sum(freqlist) for f in freqlist]


    losses = []
    for i in range(iteration):
        # Training word2vec using SGD
        while True:
            centerWord, contextWords = getRandomContext(corpus, window_size)
            if len(contextWords) == window_size * 2:
                break

        # to be implemented
        centerInd = word2ind[centerWord]
        contextInds = [word2ind[w] for w in contextWords]

        # choose whether use learning rate decay
        lr = learning_rate * (1 - i / iteration)
        if lr < 0.00001:
            lr = 0.00001

        if mode == "CBOW":
            if ns == 0:
#################################################################
# Hierarchical Softmax CBOW
#
# CBOW는 주변 단어를 통해 중심 단어를 예측
# → 따라서, nodes는 중심 단어로 가는데 거치는 허프만 트리의 노드들
#            codes는 중심 단어의 허프만 코드
#
# contextInds는 CBOW_HS 내부에서 torch.sum으로 v_wi를 구할 때 사용된다.
# W_out[nodes]는 W_out에서 업데이트할 노드들의 W_out에서 vector를 의미
#################################################################
                nodes, codes = ind2node[centerInd]
                L, G_emb, G_out = CBOW_HS(centerInd, contextInds, codes, W_emb, W_out[nodes])
                
                W_emb[contextInds] -= lr * G_emb
                W_out[nodes] -= lr * G_out
                losses.append(L.item())

            else:
#################################################################
# Negative Sampling CBOW
#
#################################################################
                samples = []
                while len(samples) < ns:
                    sample = np.random.choice(wordlist, p=problist, size=1, replace=False)[0]
                    # sample = random.choices(wordlist, weights=problist, k=1)
                    if sample not in contextInds and sample != centerInd:
                        samples.append(sample)
                
                # print("Center Word:", centerWord)
                # print("Center Word Index:", centerInd)
                # print("Context Words Indices:", contextInds)
                # print("Negative Samples Indices:", samples)

                L, G_emb, G_out = CBOW_NS(centerInd, contextInds, W_emb, W_out, samples)

                W_emb[contextInds] -= lr * G_emb
                W_out -= lr * G_out
                losses.append(L.item())

        elif mode == "SG":
            if ns == 0:
#################################################################
# Hierarchical Softmax Skip-gram
#
# Skip-gram은 중심 단어를 이용해 주변 단어를 하나씩 예측
# → 따라서 매번 nodes와 codes는 contextInd에 따라 바뀌어야 한다.
#    nodes는 주변 단어(i)로 가는데 거치는 허프만 트리의 노드들
#    codes는 주변 단어(i)의 허프만 코드
#
#  centerInd는 SG_HS 내부에서 v_wi를 얻을 때 사용된다
#################################################################
                for contextInd in contextInds:
                    nodes, codes = ind2node[contextInd]
                    L, G_emb, G_out = Skipgram_HS(centerInd, contextInd, codes, W_emb, W_out[nodes])
                    W_emb[centerInd] -= lr * G_emb.squeeze()
                    W_out[nodes] -= lr * G_out

            else:
#################################################################
# Negative Sampling Skip-gram
#
################################################################# 
                samples = []
                while len(samples) < ns:
                    sample = np.random.choice(wordlist, p=problist, size=1, replace=False)[0]
                    # sample = random.choices(wordlist, weights=problist, k=1)
                    if sample not in contextInds and sample != centerInd:
                        samples.append(sample)

                for contextInd in contextInds:
                    L, G_emb, G_out = Skipgram_NS(centerInd, contextInd, W_emb, W_out, samples)
                    W_emb[centerInd] -= lr * G_emb.squeeze()
                    W_out -= lr * G_out

            losses.append(L.item())
        else:
            print("Unkwnown mode : " + mode)
            exit()

        if i % 1000 == 0:
            avg_loss = sum(losses) / len(losses)
            print("%s : %d//%d * 1000 --- Loss : %f    lr : %f" % (mode, i / 1000, iteration / 1000, avg_loss, lr, ))
            losses = []

    return W_emb, W_out