# -*- coding: utf-8 -*-
"""BertSumTransformer.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1A3Iz-aaVzhow8vsE1Yu4iShPLPZe5VML
"""

import torch
from torch import nn
from transformers import BertModel, BertTokenizer
import json
import nltk
import pandas as pd
from konlpy.tag import Okt
import networkx as nx
nltk.download('punkt')

class BertSumTransformer(nn.Module):
    def __init__(self, bert_model_name, num_transformer_layers, nhead, dim_feedforward):
        super(BertSumTransformer, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)

        # Transformer Encoder Layer 추가
        transformer_encoder_layer = nn.TransformerEncoderLayer(d_model=self.bert.config.hidden_size, nhead=nhead, dim_feedforward=dim_feedforward)
        self.transformer_encoder = nn.TransformerEncoder(transformer_encoder_layer, num_layers=num_transformer_layers)

        # BERT 출력 차원에 맞춘 분류기
        self.classifier = nn.Linear(self.bert.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state

        # Transformer Encoder 적용
        transformer_output = self.transformer_encoder(sequence_output)
        logits = self.classifier(transformer_output).squeeze(-1)
        return logits

# 중요한 문장 추출 함수
def extract_important_sentences(text, model, tokenizer, num_sentences=2):
    model.eval()
    with torch.no_grad():
        input_ids, attention_masks, sentences = create_input(text, tokenizer)
        logits = model(input_ids, attention_masks)
        scores = torch.sigmoid(logits).squeeze().tolist()

        important_sentence_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:num_sentences]
        important_sentences = [sentences[i] for i in important_sentence_indices]
        return important_sentences

# 입력 생성 함수
def create_input(text, tokenizer):
    sentences = nltk.sent_tokenize(text)
    input_ids = []
    attention_masks = []

    for sent in sentences:
        encoded_dict = tokenizer.encode_plus(sent, add_special_tokens=True, max_length=35, pad_to_max_length=True, return_attention_mask=True, return_tensors='pt')
        input_ids.append(encoded_dict['input_ids'])
        attention_masks.append(encoded_dict['attention_mask'])

    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)

    return input_ids, attention_masks, sentences


# 모델 초기화
bert_model_name = 'bert-base-multilingual-cased'
num_transformer_layers = 2
nhead = 4
dim_feedforward = 500
sentence_model = BertSumTransformer(bert_model_name, num_transformer_layers, nhead, dim_feedforward)
tokenizer = BertTokenizer.from_pretrained(bert_model_name)