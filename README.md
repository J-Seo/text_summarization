# Deep문학도 text_summarization

---
## 0. Basic information

text summarization package with python based on deep-learning  

## File Manifest

 1) config.py: Identify your own local path according to user desktop or device
 
 2) main.py: for running total package
 
 3) model.py: LSTM model for natural language generate with sequence to sequence 
 
 4) data.py: load dataset & preprocessing the data
 
 5) data_in repo: input dataset 
 
 6) data_out repo: checkpoint & vocabulary storage


---

## 1. Prerequisites 

 1) Python 3.6.5
 2) Tensorflow 1.14.0 or Tensorflow 2.0 
 3) Pandas (install fitting on tensorflow version)
 4) Numpy (install fitting on tensorflow version)
 5) sys
 6) os

---

## 2. Authors

![logo](deepmoon.png)

Jaehyung Seo

Minhyeong Lee

Donghwan Daniel Chang

Kanghee Lee

Dongyeon Yoo

---

## 3. Sources

참고: 전창욱, 최태균, 조중현 / 위키북스 / 텐서플로와 머신러닝으로 시작하는 자연어처리 

참고: 김기헌 / 한빛 미디어 / 김기헌의 자연어 처리 딥러닝 캠프 

---

## 4. Change Log

Version 1. 2019.08.27

Version 2. 2019.09.15

Version 3. 2019.10.14 (Latest version)

---

## 5. How to use

 In terminal, "python main.py 'your own sentence'" can create headline which is answer for CSAT with learning process 
 In shell, "!python main.py 'your own sentnence" can create headline which is answer for CSAT with learning process

---

## 6. Known Issues

main.ipyb error 
Need more specific preprocessing for input dataset
