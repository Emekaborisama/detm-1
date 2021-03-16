import os
from os import path

import pandas as pd
import numpy as np
from gensim.models.word2vec import Word2Vec
import json
import pickle

def main():
    if not path.exists('word2vec.model'):
        sentences = []
        raw_data = pd.read_json("./datasets/downloaded/vaccine-forums-tokenized.json")

        for i in range(raw_data.shape[0]):
            word_lst = [word_meta['value'] for word_meta in raw_data.at[i,'tokens']]
            sentences.append(word_lst)
        
        model = Word2Vec(sentences, min_count=1)
        model.save('word2vec.model')

    if not path.exists('./datasets/processed/embedding.json'):
        model = Word2Vec.load('word2vec.model')
        data = dict()
        for word in model.wv.vocab:
            data[word] = model.wv[word].tolist()
        
        with open('./datasets/processed/embedding.json', 'w') as fp:
            json.dump(data, fp)
    
    with open('./datasets/processed/embedding.json', 'rb') as fp:
        data = json.load(fp)



if __name__ == '__main__':
    main()