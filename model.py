# -*- coding: utf-8 -*-

# LSTM 모델 사용

## 1. model 함수

## 2. make_lstm_cell 함수 (LSTM과 드롭아웃 적용)

# 2. make_lstm_cell 함수 

import tensorflow as tf
import sys

from configs import DEFINES

## mode: 작동 중인 모드, hiddenSize: 은닉 상태 벡터 값의 차원, index: 여러 개의 LSTM 중 각 스텝의 index
def make_lstm_cell(mode, hiddenSize, index):
    cell = tf.nn.rnn_cell.BasicLSTMCell(hiddenSize, name = 'lstm'+str(index))
    if mode == tf.estimator.ModeKeys.TRAIN: ## 학습 중이라면? 드롭 아웃, 아니면 꺼야함. 
        cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob = DEFINES.dropout_width)
        
    return cell

# 1. main model 함수

## features: 모델 입력 함수를 통해 만들어진 입력 값, 딕셔너리 형태, 인코더와 디코더 값으로 구성
## labels: 디코더의 타깃값
## mode: 학습? 검증? 평가?
## params: 모델에 적용되는 몇 가지 인자 값을 사전 형태로 전달 받음. 모드는 상수 값으로 저장 
def model(features, labels, mode, params):
    TRAIN = mode == tf.estimator.ModeKeys.TRAIN
    EVAL = mode == tf.estimator.ModeKeys.EVAL
    PREDICT = mode == tf.estimator.ModeKeys.PREDICT
    

## 인코더 & 디코더 구현 ==> 입력 값을 모델에 적용 가능하도록 벡터화
## 임베딩 행렬 초기화 > 원핫 인코딩 or 임베딩 벡터 만들기 > 임베딩 행렬 생성

    # 인코더 임베딩 벡터 
    ## 임베딩 벡터 만드는 경우
    if params['embedding'] == True: ## 작동에 필요한 내용을 사전 형태로 가지고 있음
        initializer = tf.contrib.layers.xavier_initializer() ## 초기화 
        embedding_encoder = tf.get_variable(name = "embedding_encoder", 
                                   shape = [params['vocabulary_length'],
                                           params['embedding_size']],
                                   dtype = tf.float32, 
                                   initializer = initializer,
                                   trainable = True)
    
    
    ## 원핫 인코딩을 사용하는 경우 
    else:
        ## 단순 단위행렬로 정의
        embedding_encoder = tf.eye(num_rows = params['vocabulary_length'], dtype = tf.float32)
        embedding_encoder = tf.get_variable(name = "embedding_encoder", 
                                   initializer = embedding_encoder,
                                   trainable = False)
    
    
    ## embedding lookup 함수를 통하여 임베딩 벡터 생성 
    #embedding_encoder = tf.nn.embedding_lookup(params = embedding, ids = features['input'])
    #embedding_decoder = tf.nn.embedding_lookup(params = embedding, ids = features['output'])
     
    ## embedding 벡터에 배치 적용
    embedding_encoder_batch = tf.nn.embedding_lookup(params = embedding_encoder, 
                                                     ids = features['input'])
    
    
    # 디코더 임베딩 벡터 
    ## 임베딩 벡터 만드는 경우
    if params['embedding'] == True: ## 작동에 필요한 내용을 사전 형태로 가지고 있음
        initializer = tf.contrib.layers.xavier_initializer() ## 초기화 
        embedding_decoder = tf.get_variable(name = "embedding_decoder", 
                                   shape = [params['vocabulary_length'],
                                           params['embedding_size']],
                                   dtype = tf.float32, 
                                   initializer = initializer,
                                   trainable = True)
    
    
    ## 원핫 인코딩을 사용하는 경우 
    else:
        ## 단순 단위행렬로 정의
        embedding_decoder = tf.eye(num_rows = params['vocabulary_length'], dtype = tf.float32)
        embedding_decoder = tf.get_variable(name = "embedding_decoder", 
                                   initializer = embedding_decoder,
                                   trainable = False)
    
    
    ## embedding lookup 함수를 통하여 임베딩 벡터 생성 
    #embedding_encoder = tf.nn.embedding_lookup(params = embedding, ids = features['input'])
    #embedding_decoder = tf.nn.embedding_lookup(params = embedding, ids = features['output'])
    
    
    ## embedding 벡터에 배치 적용
    embedding_decoder_batch = tf.nn.embedding_lookup(params = embedding_decoder, 
                                                     ids = features['output'])
    
    
    
    ## 인코더 구현
    ## variable_scope: 변수 공유 함수, 해당 변수의 범위를 encoder_scope로 명명
    with tf.variable_scope('encoder_scope', reuse = tf.AUTO_REUSE):
        ## multi layer 사용
        if params['multilayer'] == True:
            encoder_cell_list = [make_lstm_cell(mode, params['hidden_size'], i) ## 다층 LSTM 생성
                                for i in range(params['layer_size'])] ## 층 수 만큼 반복 
            rnn_cell = tf.contrib.rnn.MultiRNNCell(encoder_cell_list)
        
        ## single layer 사용    
        else:
            rnn_cell = make_lstm_cell(mode, params['hidden_size'], "") ## 하나의 LSTM 생성
        
        
        ## dynamic rnn에 적용, 임베딩 값을 입력으로 넣음.
        encoder_outputs, encoder_states = tf.nn.dynamic_rnn(cell = rnn_cell,
                                                           inputs = embedding_encoder_batch,
                                                           dtype = tf.float32)
     
    ## 디코더 구현
    ## t step의 결과 --> t+1 step의 입력으로 사용
    ## 디코더 입력 값 == 디코더 함수 결과로 나온 라벨
    with tf.variable_scope('decoder_scope', reuse = tf.AUTO_REUSE):
        ## multi layer 사용 
        if params['multilayer'] == True:
            decoder_cell_list = [make_lstm_cell(mode, params['hidden_size'], i)
                                for i in range(params['layer_size'])]
            rnn_cell = tf.contrib.rnn.MultiRNNCell(decoder_cell_list)
        else:
            rnn_cell = make_lstm_cell(mode, params['hidden_size'], "")
        
        
        ## 첫 스텝의 은닉 상태 벡터 값을 인코더의 마지막 스텝의 은닉 상태 벡터 값으로 초기화
        decoder_initial_state = encoder_states
        
        ## dynamic rnn에 적용, 임베딩 값을 입력으로 넣음.
        decoder_outputs, decoder_states = tf.nn.dynamic_rnn(cell = rnn_cell,
                                                           inputs = embedding_decoder_batch,
                                                           initial_state = decoder_initial_state,
                                                           dtype = tf.float32)
        
        
        ## Dense 층에 적용
        ## 결과값의 차원을 단어의 수로 변경 >> 최대값을 가지는 위치의 단어를 출력하도록 함
        logits = tf.keras.layers.Dense(params['vocabulary_length'])(decoder_outputs)
        predict = tf.argmax(logits, 2) ## 두 번째 차원에 최대값 인덱스가 나옴 
        
        
        ## mode == Predict일 경우, 손실값과 최적화가 필요 없음
        if PREDICT:
            predictions = {
                'indexs': predict
            }
            
            return tf.estimator.EstimatorSpec(mode, predictions = predictions)
        
        ## mode == Train or Eval일 경우, 예측값을 딕셔너리 형태로 저장 후 Estimator 객체로 리턴
        labels_ = tf.one_hot(labels, params['vocabulary_length']) ## 정답 레이블
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits = logits,
                                                                        labels = labels_))
        
        
        ## 정확도 측정
        accuracy = tf.metrics.accuracy(labels = labels, predictions = predict, name = 'acc0p')
        metrics = {'accuracy': accuracy}
        tf.summary.scalar('accuracy', accuracy[1])
        
        
        ## mode == Eval 이라면?
        ## 예측모드와 동일하지만, loss값과 accuracy 값을 전달해야 함
        if EVAL:
            return tf.estimator.EstimatorSpec(mode, loss = loss, eval_metric_ops = metrics)
        
        ## mode == Train 이라면?
        ## 최적화가 필요함
        assert TRAIN
        
        optimizer = tf.train.AdamOptimizer(learning_rate = DEFINES.learning_rate)
        train_op = optimizer.minimize(loss, global_step = tf.train.get_global_step())
        
        return tf.estimator.EstimatorSpec(mode, loss = loss, train_op = train_op)
