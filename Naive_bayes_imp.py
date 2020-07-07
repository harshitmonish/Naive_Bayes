# -*- coding: utf-8 -*-
"""
Created on Sat Jun 13 17:36:36 2020

@author: harshitm
"""

import numpy as np
import nltk 
import re
import math
from random import choice
import time
import _pickle as cpickle
import os

def get_data(x_file, y_file):
    x_file_obj = open(x_file, 'r', encoding="utf8")
    x = []
    for line in x_file_obj:
        x.append(line)
    y = np.loadtxt(y_file)
    return x, y

def get_word_counts(words):
    word_counts = {}
    for word in words:
        word_counts[word] = word_counts.get(word, 0.0) + 1.0
    return word_counts
        
def train_model(x, y, modi, use_bigrams):
    vocab_dict = {}
    word_count_labels = {}
    theta_words = {}
    len_of_examples = {}
    labels_count = {}
    log_labels_prior = {}
    if(modi):
        file_train_name = "objs/train_obj_modi"
    else:
        file_train_name = "objs/train_obj"
    if os.path.exists(file_train_name):
        fileObj = open(file_train_name, 'rb')
        arr = cpickle.load(fileObj)
        log_labels_prior = arr[0] 
        theta_words = arr[1]
        vocab_dict = arr[2]
        labels_count = arr[3]
        return log_labels_prior, theta_words, vocab_dict, labels_count
    
    N1 = len(y)
    for i in range(1,11):
        if i == 5 or i == 6: continue
        word_count_labels[i] = {}
        theta_words[i] = {}
        len_of_examples[i] = 0

    i = 0
    for line in x:
        #tokens = nltk.word_tokenize(line)
        line = line.lower()
        tokens = re.split("\W+", line)
        if(use_bigrams == 1):
            bigrams = [' '.join(b) for b in zip(tokens[:-1], tokens[1:])]
            tokens = bigrams
        word_count = get_word_counts(tokens)
        for word, count in word_count.items():
            if word not in vocab_dict:
                vocab_dict[word] = 0
            if word not in word_count_labels[y[i]]:
                word_count_labels[y[i]][word] = 0.0
            word_count_labels[y[i]][word] += count
        if y[i] not in labels_count:
            labels_count[y[i]] = 0.0
        labels_count[y[i]] += 1
        len_of_examples[y[i]] += len(tokens)
        i+=1
        
    for j in range(1,11):
        if j ==5 or j == 6: continue
        log_labels_prior[j] = math.log(labels_count[j] / N1)
        
    for j in range(1,11):
        if j == 5 or j == 6: continue
        for word in vocab_dict:
           temp = (int(word_count_labels[j].get(word,0.0)) + 1) / (int(len_of_examples[j]) + int(len(vocab_dict)))
           theta_words[j][word] = temp   
    fileObj = open(file_train_name,'wb')    
    arr = [log_labels_prior, theta_words, vocab_dict, labels_count]     
    cpickle.dump(arr, fileObj)
    return log_labels_prior, theta_words, vocab_dict, labels_count

""" accuracy = 12.48% """
def classify_random(x, modi, train):
    if(modi):
        if(train):
            file_r_c_name = "objs/r_c_name_modi_train"
        else:
            file_r_c_name = "objs/r_c_name_modi_test"
    else:
        if(train):
            file_r_c_name = "objs/r_c_name_train"
        else:
            file_r_c_name = "objs/r_c_name_test"
    if os.path.exists(file_r_c_name):
        fileObj = open(file_r_c_name, 'rb')
        result = cpickle.load(fileObj)
        return result
    result = []
    for line in x:
        result.append(choice([i for i in range(1,11) if i not in [5,6]]))
    fileObj = open(file_r_c_name,'wb')    
    cpickle.dump(result, fileObj) 
    return result

""" accuracy = 20.09% """
def classify_majority(x, labels_count, modi, train):
    if(modi):
        if(train):
            file_mj_c_name = "objs/mj_c_name_modi_train"
        else:
            file_mj_c_name = "objs/mj_c_name_modi_test"
    else:
        if(train):
            file_mj_c_name = "objs/mj_c_name_train"
        else:
            file_mj_c_name = "objs/mj_c_name_test"
    if os.path.exists(file_mj_c_name):
        fileObj = open(file_mj_c_name, 'rb')
        result = cpickle.load(fileObj)
        return result
    result = []
    for line in x:
        result.append(max(labels_count, key = labels_count.get))
        
    fileObj = open(file_mj_c_name,'wb')    
    cpickle.dump(result, fileObj)  
    return result

""" 
with bigrams:
    accuracy:
    train = 99.0%
    test = 38.8%
without bigrams:
    accuracy:
    train = 70.15%
    test = 39.5% 

"""    
def naive_bayes_classifier(x, log_labels_prior, theta_words, vocab_dict, modi, train, use_bigrams):
    if(modi):
        if(train):
            file_nv_c_name = "objs/nv_c_res_modi_train"
        else:
            file_nv_c_name = "objs/nv_c_res_modi_test"
    else:
        if(train):
            file_nv_c_name = "objs/nv_c_res_train"
        else:
            file_nv_c_name = "objs/nv_c_res_test"
    if os.path.exists(file_nv_c_name):
        fileObj = open(file_nv_c_name, 'rb')
        result = cpickle.load(fileObj)
        return result
    
    result = []
    for line in x:
        #tokens = nltk.word_tokenize(line)
        #line = line.lower()
        tokens = re.split("\W+", line)
        if(use_bigrams):
            bigrams = [' '.join(b) for b in zip(tokens[:-1], tokens[1:])]
            tokens = bigrams
        class_label_score = {1:0, 2:0, 3:0, 4:0, 7:0, 8:0, 9:0, 10:0}
        word_count = get_word_counts(tokens)
        for word, count in word_count.items():
            if word not in vocab_dict: continue
            for i in range(1,11):
                if i== 5 or i == 6: continue
                class_label_score[i] += (math.log(theta_words[i][word]))
            
        for i in range(1,11):
            if i == 5 or i == 6:continue
            class_label_score[i] += float((log_labels_prior[i]))
        #print(class_label_score)
        result.append(max(class_label_score, key = class_label_score.get))
        
    fileObj = open(file_nv_c_name,'wb')    
    cpickle.dump(result, fileObj)    
    return result

def prediction(x, log_labels_prior, theta_words, vocab_dict, labels_count, classifier, modi, train, use_bigrams):
    if classifier == 1:
        result = classify_random(x, modi, train)
    elif classifier == 2:
        result = classify_majority(x, labels_count, modi, train)
    elif classifier == 3:
        result = naive_bayes_classifier(x, log_labels_prior, theta_words, vocab_dict, modi, train, use_bigrams)

    return np.array(result)

def find_accuracy(y, result):
    N = len(y)
    sum = 0
    for i in range(0,N):
        if y[i] == result[i]:
            sum += 1
    return ((sum/N)*100)
    
def create_conf_matrix(expected, predicted):
    conf_mat = np.zeros((11,11), dtype = int)
    N = len(expected)
    for i in range(0,N):
        if int(expected[i]) == 5 or int(expected[i]) == 6: continue
        conf_mat[int(expected[i])][int(predicted[i])] +=1
    
    conf_mat = np.delete(conf_mat, 0, 0)
    conf_mat = np.delete(conf_mat, 4, 0)
    conf_mat = np.delete(conf_mat, 4, 0)
    conf_mat = np.delete(conf_mat, 0, 1)
    conf_mat = np.delete(conf_mat, 4, 1)
    conf_mat = np.delete(conf_mat, 4, 1)
    return conf_mat
    
def main():
    x_train_file = "./datasets/imdb_mod/imdb_train_text.txt"
    y_train_file = "./datasets/imdb/imdb_train_labels.txt"
    x_test_file = "./datasets/imdb_mod/imdb_test_text.txt"
    y_test_file = "./datasets/imdb/imdb_test_labels.txt"
    start_time = time.time()
    X_train, Y_train = get_data(x_train_file, y_train_file)
    X_test, Y_test = get_data(x_test_file, y_test_file)
    log_labels_prior, theta_words, vocab_dict, labels_count = train_model(X_train, Y_train, 1, 0)
    """
    classifier:
    1.-> random classifier
    2. -> majority classifier
    3. -> naive bayes classifier
    """
    result = prediction(X_test, log_labels_prior,theta_words, vocab_dict, labels_count, 3, 0, 0, 0) #classifier, modi, train
    accuracy = find_accuracy(Y_test, result)
    print(("Accuracy : {0:.2f}%".format(accuracy)))
    print("--- %s seconds ---" % (time.time() - start_time))
    conf_mat = create_conf_matrix(result , Y_train)
    print("\n Confusion Matrix : \n")
    print(conf_mat)
    
if __name__ == "__main__":
    main()
