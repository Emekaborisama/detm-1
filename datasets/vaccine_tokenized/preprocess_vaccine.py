import os
from os import path
import operator
import json

import pandas as pd
import numpy as np
import spacy
from gensim.models.word2vec import Word2Vec

MAX_WEEKNUM = 40
MIN_WEEKNUM = 10
WEEKNUM = 24

nlp = spacy.load('en_core_web_lg')

# Print iterations progress
def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
    # Print New Line on Complete
    if iteration == total: 
        print()

def main():
    source = "./datasets/downloaded/vaccine-forums-tokenized.json"
    target = "./datasets/processed/vaccine-forums-processed.json"
    process_data(source, target)

    data = pd.read_json(target)

    with open('./datasets/processed/vocab.json', 'rb') as fp:
        vocab = json.load(fp)

    new_vocab = dict()
    model = Word2Vec.load('word2vec.model')

    idx = 0
    for i in range(data.shape[0]):
        tokens = data.at[i, 'tokens']
        new_tokens = []

        for t in tokens:
            if vocab[t] not in new_vocab:
                new_vocab[vocab[t]] = idx
                idx += 1
            new_tokens.append(new_vocab[vocab[t]])
        data.at[i, 'tokens'] = new_tokens
    
    vocab = []
    sorted_dictionary = sorted(new_vocab.items(), key=operator.itemgetter(1))
    for item in sorted_dictionary:
        vocab.append(item[0])
    
    data = data.to_json("./datasets/processed/vaccine-forums-newprocessed.json", orient="columns")

    with open('./datasets/processed/new_vocab.json', 'w') as fp:
        json.dump(vocab, fp)

    return

def process_data(source, target):
    if not path.exists(source):
        print("Please download the dataset vaccine-forums-tokenized.json")
        return
    
    if not path.exists(target):
        print("<--- Preprocessing the data --->")

        raw_data = pd.read_json("./datasets/downloaded/vaccine-forums-tokenized.json")
        raw_data = raw_data.filter(items=['post_updated_at', 'cleaned_post_text'])
        raw_data['counts'] = [[] for i in range(raw_data.shape[0])]
        raw_data['tokens'] = [[] for i in range(raw_data.shape[0])]

        printProgressBar(0, raw_data.shape[0], prefix='Progress:', suffix='Complete', length=15)

        idx = 0
        dictionary = dict()

        for i in range(raw_data.shape[0]):
            paragraph = raw_data.at[i, 'cleaned_post_text']
            curr_doc = nlp(paragraph)

            word_lst = [token.lemma_.lower() for token in curr_doc if not token.is_stop and not token.is_punct and not token.is_space]
            raw_data.at[i, 'tokens'], raw_data.at[i, 'counts'] = np.unique(word_lst, return_counts=True)

            # divided into weeks
            time = raw_data.at[i, 'post_updated_at'].week
            if time >= 40:
                time -= 40
            else:
                time += 13
            raw_data.at[i, 'post_updated_at'] = time
            
            word_idx = []
            for word in raw_data.at[i, 'tokens']:
                if word not in dictionary:
                    dictionary[word] = idx
                    idx += 1
                word_idx.append(dictionary[word])
            
            raw_data.at[i, 'tokens'] = word_idx

            printProgressBar(i+1, raw_data.shape[0], prefix='Progress:', suffix='Complete', length=15)

        vocab = []
        sorted_dictionary = sorted(dictionary.items(), key=operator.itemgetter(1))
        for item in sorted_dictionary:
            vocab.append(item[0])
        
        raw_data = raw_data.filter(items=['post_updated_at', 'tokens', 'counts'])
        data = raw_data.to_json(target, orient="columns")

        with open('./datasets/processed/vocab.json', 'w') as fp:
            json.dump(vocab, fp)
        
        print("<--- Finished --->")
    
    data = pd.read_json(target)
    
    return data

# def process_data(source, target):
#     if not path.exists(source):
#         print("Please download the dataset vaccine-forums-tokenized.json")
    
#     if not path.exists(target):
#         raw_data = pd.read_json("./datasets/downloaded/vaccine-forums-tokenized.json")
#         raw_data = raw_data.filter(items=['post_updated_at', 'tokens'])
#         raw_data['counts'] = [[] for i in range(raw_data.shape[0])]

#         idx = 0
#         dictionary = dict()

#         for i in range(raw_data.shape[0]):
#             word_lst = [word_meta['value'] for word_meta in raw_data.at[i,'tokens']]
#             raw_data.at[i, 'tokens'], raw_data.at[i, 'counts'] = np.unique(word_lst, return_counts=True)
#             time = raw_data.at[i, 'post_updated_at'].timestamp()
#             # if time > (MAX_TIMESTAMP - MIN_TIMESTAMP) / 2 + MIN_TIMESTAMP:
#             #     time = 1
#             # else:
#             #     time = 0
#             time = int((time - MIN_TIMESTAMP) / ((MAX_TIMESTAMP - MIN_TIMESTAMP) / NUM_INTERVALS))
#             raw_data.at[i, 'post_updated_at'] = time
            
#             word_idx = []
#             for word in raw_data.at[i, 'tokens']:
#                 if word not in dictionary:
#                     dictionary[word] = idx
#                     idx += 1
#                 word_idx.append(dictionary[word])
            
#             raw_data.at[i, 'tokens'] = word_idx

#         vocab = []
#         sorted_dictionary = sorted(dictionary.items(), key=operator.itemgetter(1))
#         for item in sorted_dictionary:
#             vocab.append(item[0])
        
#         data = raw_data.to_json(target, orient="columns")

#         with open('./datasets/processed/vocab.json', 'w') as fp:
#             json.dump(vocab, fp)

if __name__ == "__main__":
    main()