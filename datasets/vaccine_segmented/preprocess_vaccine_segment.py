import os
from os import path
import operator
import json

import pandas as pd
import numpy as np
import spacy
from gensim.models.word2vec import Word2Vec

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

MAX_WEEKNUM = 9
MIN_WEEKNUM = 2
WEEKNUM = 6

nlp = spacy.load('en_core_web_lg')

def main():
    source = "./datasets/downloaded/vaccine-merged-segmented.json"
    target = "./datasets/processed/vaccine-merged-processed.json"
    process_data(source, target)

    # data = pd.read_json(target)

    # with open('./datasets/processed/vocab.json', 'rb') as fp:
    #     vocab = json.load(fp)

    # new_vocab = dict()
    # model = Word2Vec.load('word2vec.model')

    # idx = 0
    # for i in range(data.shape[0]):
    #     tokens = data.at[i, 'tokens']
    #     new_tokens = []

    #     for t in tokens:
    #         if vocab[t] not in new_vocab:
    #             new_vocab[vocab[t]] = idx
    #             idx += 1
    #         new_tokens.append(new_vocab[vocab[t]])
    #     data.at[i, 'tokens'] = new_tokens
    
    # vocab = []
    # sorted_dictionary = sorted(new_vocab.items(), key=operator.itemgetter(1))
    # for item in sorted_dictionary:
    #     vocab.append(item[0])
    
    # data = data.to_json("./datasets/processed/vaccine-forums-newprocessed.json", orient="columns")

    # with open('./datasets/processed/new_vocab.json', 'w') as fp:
    #     json.dump(vocab, fp)

    return

def process_data(source, target):
    if not path.exists(source):
        print("Please download the dataset vaccine-merged-segmented.json")
        return
    
    if not path.exists(target):
        print("<--- Preprocessing the data --->")

        raw_data = pd.read_json(source)

        printProgressBar(0, raw_data.shape[0], prefix='Progress:', suffix='Complete', length=15)

        idx = 0
        dictionary = dict()
        tokens = []
        counts = []
        times = []

        for i in range(raw_data.shape[0]):
            sentence = raw_data.at[i, 'tokens']

            if not sentence:
                continue

            word_lst = [token['value'] for token in sentence]
            curr_token, curr_count = np.unique(word_lst, return_counts=True)

            # divided into months
            time = raw_data.at[i, 'date'].month
            if time >= 9:
                time -= 9
            else:
                time += 3
            
            word_idx = []
            for word in curr_token:
                if word not in dictionary:
                    dictionary[word] = idx
                    idx += 1
                word_idx.append(dictionary[word])
            
            tokens.append(word_idx)
            counts.append(curr_count)
            times.append(time)

            printProgressBar(i+1, raw_data.shape[0], prefix='Progress:', suffix='Complete', length=15)

        vocab = []
        sorted_dictionary = sorted(dictionary.items(), key=operator.itemgetter(1))
        for item in sorted_dictionary:
            vocab.append(item[0])
        
        data = dict()
        data['post_updated_at'] = times
        data['tokens'] = tokens
        data['counts'] = counts
        data = pd.DataFrame(data)
        data.to_json(target, orient="columns")
        data = pd.read_json(target)

        with open('./datasets/processed/vocab.json', 'w') as fp:
            json.dump(vocab, fp)
        
        print("<--- Finished --->")
    
    data = pd.read_json(target)
    
    return data

if __name__ == "__main__":
    main()