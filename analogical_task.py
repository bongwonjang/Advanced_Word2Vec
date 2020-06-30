import numpy as np
import torch 

def analogical_task(word2ind, ind2word, word_matrix):
    ART = [
        ['brother', 'sister', 'grandson', 'granddaughter'],
        ['apparent', 'apparently', 'rapid', 'rapidly'],
        ['possibly', 'impossibly', 'ethical', 'unethical'],
        ['great', 'greater', 'tough', 'tougher'],
        ['easy', 'easiest', 'lucky', 'luckiest'],
        ['think', 'thinking', 'read', 'reading'],
        ['walking', 'walked', 'swimming', 'swam'],
        ['mouse', 'mice', 'dollar', 'dollars'],
        ['work', 'works', 'read', 'reads']
        ]

    problem_vec = []
    for relation in ART:
        idx_0 = word2ind[relation[0]]
        idx_1 = word2ind[relation[1]]
        idx_2 = word2ind[relation[2]]
        idx_3 = word2ind[relation[3]]
  
        problem_vec.append([relation[3], word_matrix[idx_1] - word_matrix[idx_0] + word_matrix[idx_2]])
        problem_vec.append([relation[2], word_matrix[idx_0] - word_matrix[idx_1] + word_matrix[idx_3]])
        problem_vec.append([relation[1], word_matrix[idx_3] - word_matrix[idx_2] + word_matrix[idx_0]])
        problem_vec.append([relation[0], word_matrix[idx_2] - word_matrix[idx_3] + word_matrix[idx_1]])

    print("number of questions ( 36 ) ", len(problem_vec))
    print("Task START")
    for question in problem_vec:
        most_similar(question, word2ind, ind2word, word_matrix)
    print("DONE_TASK")

def regularize(query):
    # query should be [N, M] matrix
    L2_norm_query = torch.norm(query, dim=1).view(-1, 1)
    return torch.div(query, L2_norm_query)

def most_similar(query, word2ind, ind2word, word_matrix, top=5):
    ##############################################
    # 이 부분 try~catch를 실행해봐야 하지 않을까 생각 됨..
    ##############################################
    if query[0] not in word2ind:
        print('%s (을)를 찾을 수 없습니다.' % query)
        return

    print('\n[answer is] ' + query[0])

    reg_query_vec = regularize(query[1].view(1, -1)) # [300, ] -> [1, 300]
    reg_word_matrix = regularize(word_matrix)

    similarity = torch.mm(reg_word_matrix, reg_query_vec.view(-1, 1)).view(-1).numpy()

    count = 0
    for i in (-1 * similarity).argsort():
        if type(list(ind2word.keys())[0]) == str:  
            print(ind2word[str(i)]) # from loaded file
        else:
            print(ind2word[i])      # just after training
        count += 1
        if count >= top:
            return