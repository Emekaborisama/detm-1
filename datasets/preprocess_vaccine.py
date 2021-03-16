import os
from os import path
import operator
import json

import pandas as pd
import numpy as np

MAX_TIMESTAMP = 1615295491.0
MIN_TIMESTAMP = 1601267365.0
NUM_INTERVALS = 30

def main():
    source = "./datasets/downloaded/vaccine-forums-tokenized.json"
    target = "./datasets/processed/vaccine-forums-processed.json"
    process_data(source, target)

    data = pd.read_json(target)

    with open('./datasets/processed/vocab.json', 'rb') as fp:
        vocab = json.load(fp)
    
    ma = 0
    mi = 10
    for i in range(data.shape[0]):
        ma = max(ma, data.at[i, 'post_updated_at'])
        mi = min(mi, data.at[i, 'post_updated_at'])
    print(ma, mi)


def process_data(source, target):
    if not path.exists(source):
        print("Please download the dataset vaccine-forums-tokenized.json")
    
    if not path.exists(target):
        raw_data = pd.read_json("./datasets/downloaded/vaccine-forums-tokenized.json")
        raw_data = raw_data.filter(items=['post_updated_at', 'tokens'])
        raw_data['counts'] = [[] for i in range(raw_data.shape[0])]

        idx = 0
        dictionary = dict()

        for i in range(raw_data.shape[0]):
            word_lst = [word_meta['value'] for word_meta in raw_data.at[i,'tokens']]
            raw_data.at[i, 'tokens'], raw_data.at[i, 'counts'] = np.unique(word_lst, return_counts=True)
            time = raw_data.at[i, 'post_updated_at'].timestamp()
            # if time > (MAX_TIMESTAMP - MIN_TIMESTAMP) / 2 + MIN_TIMESTAMP:
            #     time = 1
            # else:
            #     time = 0
            time = int((time - MIN_TIMESTAMP) / ((MAX_TIMESTAMP - MIN_TIMESTAMP) / NUM_INTERVALS))
            raw_data.at[i, 'post_updated_at'] = time
            
            word_idx = []
            for word in raw_data.at[i, 'tokens']:
                if word not in dictionary:
                    dictionary[word] = idx
                    idx += 1
                word_idx.append(dictionary[word])
            
            raw_data.at[i, 'tokens'] = word_idx

        vocab = []
        sorted_dictionary = sorted(dictionary.items(), key=operator.itemgetter(1))
        for item in sorted_dictionary:
            vocab.append(item[0])
        
        data = raw_data.to_json(target, orient="columns")

        with open('./datasets/processed/vocab.json', 'w') as fp:
            json.dump(vocab, fp)

if __name__ == "__main__":
    main()