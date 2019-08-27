# -*- coding: utf-8 -*-

from configs import DEFINES
import tensorflow as tf
import re 
import os
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import enum

FILTERS = '[-=+,#/\?:^$.@*\"※~&%ㆍ!』\\‘|\(\)\[\]\<\>`\'…》]'
PAD = "<PADDING>"
STD = "<START>"
END = "<END>"
UNK = "<UNKNOWN>"

PAD_INDEX = 0
STD_INDEX = 1
END_INDEX = 2
UNK_INDEX = 3

MARKER = [PAD, STD, END, UNK]
CHANGE_FILTER = re.compile(FILTERS)


## 데이터 불러오기
def load_data():
    data_df = pd.read_csv(DEFINES.data_path, header = 0)
    content, title = list(data_df['content']), list(data_df['title'])
    train_input, eval_input, train_label, eval_label = train_test_split(content, title, 
                                                                       test_size = 0.33, random_state = 42)
    return train_input, train_label, eval_input, eval_label

    
    

## 인코더 및 디코더 전처리

def enc_processing(value, dictionary):
    sequences_input_index = []
    sequences_length = []
    
    for sequence in value:
        sequence = str(sequence)
        sequence = re.sub(CHANGE_FILTER, "", sequence)
        sequence_index = []
        
        for word in sequence.split():
            if dictionary.get(word) is not None:
                sequence_index.extend([dictionary[word]])
            else:
                sequence_index.extend([dictionary[UNK]])
                
        if len(sequence_index) > DEFINES.max_sequence_length:
            sequence_index = sequence_index[:DEFINES.max_sequence_length]
            
        sequences_length.append(len(sequence_index))
        sequence_index += (DEFINES.max_sequence_length - len(sequence_index)) * [dictionary[PAD]]
        sequences_input_index.append(sequence_index)
        
        
    return np.asarray(sequences_input_index), sequences_length

def dec_input_processing(value, dictionary):
    sequences_output_index = []
    sequences_length = []
    
    for sequence in value:
        sequence = re.sub(CHANGE_FILTER, "", sequence)
        sequence_index = []
        sequence = str(sequence)
        
        ## 시작 지점에 <STD> 추가
        sequence_index = [dictionary[STD]] + [dictionary[word] for word in sequence.split()]
                
        if len(sequence_index) >= DEFINES.max_sequence_length:
            sequence_index = sequence_index[:DEFINES.max_sequence_length]
            
        sequences_length.append(len(sequence_index))
        sequence_index += (DEFINES.max_sequence_length - len(sequence_index)) * [dictionary[PAD]]
        sequences_output_index.append(sequence_index)
        
        
    return np.asarray(sequences_output_index), sequences_length

def dec_target_processing(value, dictionary):
    sequences_target_index = []
   
    for sequence in value:
        sequence = re.sub(CHANGE_FILTER, "", sequence)
        sequence_index = [dictionary[word] for word in sequence.split()]
        
        ## 마지막 지점에 <END> 추가
        if len(sequence_index) >= DEFINES.max_sequence_length:
            sequence_index = sequence_index[:DEFINES.max_sequence_length - 1] + [dictionary[END]]
        
        else:
            sequence_index += [dictionary[END]]
       
        sequence_index += (DEFINES.max_sequence_length - len(sequence_index)) * [dictionary[PAD]]
        sequences_target_index.append(sequence_index)
        
        
    return np.asarray(sequences_target_index)

## 인덱스를 문장으로 바꾸기

def pred2string(value, dictionary):
    sentence_string = []
    for v in value:
        ## 문자열 문장으로 바꾸기
        sentence_string = [dictionary[index] for index in v['indexs']]
        print(sentence_string)
        answer = ""
        
    for word in sentence_string:
        if word not in PAD and word not in END:
            answer += word
            answer += " " # 어절 단위 분할
                
    print(answer)
    return answer

## 인덱스 사전 만들기

## 데이터 전처리 후 단어 리스트 생성함수

def data_tokenizer(data):
    words =[]
    for sentence in data:
        sentence = str(sentence)
        sentence = re.sub(CHANGE_FILTER,"", sentence)
        for word in sentence.split():
            words.append(word)
    
    ## 특수 문자 제거 이후, 모든 단어를 포함하는 단어리스트 생성
    return [word for word in words if word]


def load_vocabulary():
    vocabulary_list = []
    # 없을 경우 만들기
    if (not (os.path.exists(DEFINES.vocabulary_path))):
        if (os.path.exists(DEFINES.data_path)):
            data_df = pd.read_csv(DEFINES.data_path, encoding = 'utf-8')
            content, title = list(data_df['content']), list(data_df['title'])
            
            data = []
            data.extend(content)
            data.extend(title)
            words = data_tokenizer(data)
            words = list(set(words))
            words[:0] = MARKER ## 사전에 정의한 토큰들을 제일 앞에 넣어주기
        ## 파일 만들기 
        with open(DEFINES.vocabulary_path, 'w', encoding = 'utf-8', errors= 'ignore') as vocabulary_file:
            for word in words:
                vocabulary_file.write(word + '\n')
                
    ## 생성된 파일 또는 원래 있던 기존 파일 불러오기             
    with open(DEFINES.vocabulary_path, 'r', encoding = 'utf-8', errors = 'ignore') as vocabulary_file:
        for line in vocabulary_file:
            vocabulary_list.append(line.strip())
            
    word2idx, idx2word = make_vocabulary(vocabulary_list)
    
    return word2idx, idx2word, len(word2idx)

# word2idx 및 idx2word 정의
def make_vocabulary(vocabulary_list):
    word2idx = {word:idx for idx, word in enumerate(vocabulary_list)}
    idx2word = {idx: word for idx, word in enumerate(vocabulary_list)}
    return word2idx, idx2word

# 텐서 플로에 넣기 위한 입력 데이터로 만드는 함수

## Estimator 활용

def train_input_fn(train_input_enc, train_output_dec, train_target_dec, batch_size):
    dataset = tf.data.Dataset.from_tensor_slices((train_input_enc, train_output_dec,
                                                  train_target_dec))
    dataset = dataset.shuffle(buffer_size = len(train_input_enc))
    assert batch_size is not None, "train batchSize must not be None" ## batch_size가 없을 경우 출력
    
    dataset = dataset.batch(batch_size)
    dataset = dataset.map(rearrange) ## 따로 정의해야 함
    dataset = dataset.repeat() ## 반복 수행하도록 함, 직접 멈추어야 함 
    iterator = dataset.make_one_shot_iterator()
    return iterator.get_next()


def eval_input_fn(eval_input_enc, eval_output_dec, eval_target_dec, batch_size):
    dataset = tf.data.Dataset.from_tensor_slices((eval_input_enc, eval_output_dec, 
                                                 eval_target_dec))
    
    dataset = dataset.shuffle(buffer_size = len(eval_input_enc))
    assert batch_size is not None, "eval batchSize must not be None"
    dataset = dataset.batch(batch_size)
    dataset = dataset.map(rearrange)
    dataset = dataset.repeat(1) ## 1회만 진행 
    iterator = dataset.make_one_shot_iterator()
    return iterator.get_next()

## 2개 데이터는 각각 인코더와 디코더, target은 라벨
## 3개의 인자, 2개의 입력값은 하나의 딕셔너리로 묶음.
def rearrange(input, output, target):
    features = {'input': input, "output": output}
    return features, target ### 위의 정의된 형태로 map함수를 통해서 반복 시행
