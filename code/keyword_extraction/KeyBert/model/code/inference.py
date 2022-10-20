from sentence_transformers import SentenceTransformer, util
import os
import csv
import pickle
import time
import numpy as np
import logging
import json
from keybert import KeyBERT
import torch
# model_name = 'quora-distilbert-multilingual'
model_name = 'all-mpnet-base-v2'
top_k_hits = 10    
keyphrase_ngram_range=(2,4)
# model_name = 'all-MiniLM-L12-v2'
# model = SentenceTransformer(model_name)
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def model_fn(model_dir):
    device = get_device()
    model = KeyBERT(model=model_name)

    return model

# def input_fn(json_request_data, content_type='application/json'):  
#     input_data = json.loads(json_request_data)
#     text_to_summarize = input_data['text']
#     return text_to_summarize
def input_fn(request_body, request_content_type):
    """
    Deserialize and prepare the prediction input
    """
    
    if request_content_type == "application/json":
        request = json.loads(request_body)
    else:
        request = request_body

    return request


def predict_fn(input_data, model):
    keywords = model.extract_keywords(input_data, keyphrase_ngram_range=keyphrase_ngram_range, stop_words='english', top_n=top_k_hits)
    
    hits = []
    for keyword, score in keywords:
        hits.append({'corpus_sentence': keyword, 'score': score})

    return hits


def get_device():
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    return device

# def output_fn(embedding, accept='application/json'):
#     return json.dumps({'features':embedding}, cls=NpEncoder), accept
def output_fn(prediction, response_content_type):
    """
    Serialize and prepare the prediction output
    """
    
    if response_content_type == "application/json":
        response = str(prediction)
    else:
        response = str(prediction)

    return response

if __name__ == '__main__':

    input_data = 'sdsd sdsd'
    model = model_fn('../')
    result = predict_fn(input_data, model)
    print(json.dumps({'features': result}, cls=NpEncoder))
    print(len(result))
    print(result)