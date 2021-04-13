import os
from os import path
import re

import pandas as pd
import numpy as np
from gensim.models.word2vec import Word2Vec
import json
import pickle
import spacy

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
    if not path.exists('word2vec.model'):
        print("Start Training Word2Vec Model")

        raw_data = pd.read_json('./datasets/downloaded/vaccine-merged-segmented.json')
        sentences = raw_data['tokens'].tolist()

        for i in range(len(sentences)):
            sentences[i] = [d['value'] for d in sentences[i]]

        model = Word2Vec(sentences, min_count=1, sg=1)
        model.train(sentences, total_examples=len(sentences), epochs=95)
        model.save('word2vec.model')

        print("Finish Training")
    
    if not path.exists('./datasets/processed/embedding.json'):
        print("Start Building Dictionary for Word2Vec")

        with open('./datasets/processed/vocab.json', 'rb') as fp:
            vocab = json.load(fp)

        odds = set()

        model = Word2Vec.load('word2vec.model')
        data = dict()
        for word in vocab:
            if word in model.wv.vocab:
                data[word] = model.wv[word].tolist()
            else:
                words = word.split()
                curr_vec = np.zeros((100,))
                for w in words:
                    if w in model.wv.vocab:
                        curr_vec += model.wv[w]
                    else:
                        odds.add(word)
                        break
                curr_vec /= len(words)
                data[word] = curr_vec.tolist()

        with open('./datasets/processed/embedding.json', 'w') as fp:
            json.dump(data, fp)
        
        print("Finish Building")
    
    with open('./datasets/processed/embedding.json', 'rb') as fp:
        data = json.load(fp)
    
    return

        # clean dirty words not exists in word embedding
        # raw_data = pd.read_json("./datasets/processed/vaccine-forums-processed.json")
        # for i in range(raw_data.shape[0]):
        #     is_clean = True
        #     curr_tokens = raw_data.at[i, 'tokens']
        #     curr_counts = raw_data.at[i, 'counts']
        #     while True:
        #         is_clean = True
        #         for idx in indices:
        #             if idx in curr_tokens:
        #                 is_clean = False

        #                 curr_idx = curr_tokens.index(idx)
        #                 curr_tokens.pop(curr_idx)
        #                 curr_counts.pop(curr_idx)

        #                 break
        #         if is_clean:
        #             break
        #     raw_data.at[i, 'tokens'] = curr_tokens
        #     raw_data.at[i, 'counts'] = curr_counts

if __name__ == '__main__':
    main()