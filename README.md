# Hate-Speech-Detection-using-Bidirectional-LSTM


## Overview

This project classifies tweets into three categories:

0. Hate Speech.
1. Offensive Language.
2. Neither.

Three models were built, trained, and compared:

1. Simple RNN.
2. LSTM.
3. Bidirectional LSTM.

## Dataset

Davidson Hate Speech Dataset.
around 25,000 labelled tweets.
Heavily imbalanced towards "Offensive Language" class.


## Data Pipeline

1. *Text Cleaning*: lowercasing, URL/HTML Tag removal, punctuation stripping, lemmatization, and stopword removal.
2. *Tokenization & Padding*: Keras Tokenizer with vocab size of 7000, sequences padding and truncated to length 40.
3. *Class Imbalance Handling*: Combination of RandomOverSampling for minority classes and RandomUnderSampling for the majority class, with addition of class weights.
4. *Model Training*: embedding layer trained from scratch, EarlyStopping and ReduceLROnPlateau callbacks.
5. *Evaluation*: Precision, Recall, F1-Score, and confusion matrix for every model.
6. *Deployment*: Chose the BiLSTM Model for deployment using streamlit.


## Results

| Class | Label | Precision | Recall | F1-score | Support |
|---|---|---|---|---|---|
| 0 | Hate Speech | 0.31 | 0.62 | 0.41 | 286 |
| 1 | Offensive Language | 0.96 | 0.86 | 0.91 | 3,838 |
| 2 | Neither | 0.80 | 0.91 | 0.85 | 833 |
| | **Accuracy** | | | **0.85** | 4,957 |
| | **Macro avg** | 0.69 | 0.79 | 0.72 | 4,957 |
| | **Weighted avg** | 0.89 | 0.85 | 0.87 | 4,957 |

**Numbers can change, this is only one singular instance of training and testing but other tries should give similar outputs**


## Project Structure

- Hate_Speech_Detection.ipynb #Full training pipeline
- app.py #Streamlit web application
- bilstm_hate_speech.keras #Train BiLSTM model
- tokenizer.pkl #Fitted Keras Tokenizer
- requirements.txt #Python Dependecies

## Notes:

This application is made for educational purposes, It should not be used in automated banning systems.


## Made by : NoorAldeen Faruq Al-Hara
