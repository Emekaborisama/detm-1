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
    if not path.exists('./datasets/processed/sentences.json'):
        print("Start Parsing Posts into Sentences")

        sentences = []
        raw_data = pd.read_json("./datasets/downloaded/vaccine-forums-tokenized.json")
        raw_data = raw_data.filter(items=['cleaned_post_text'])

        printProgressBar(0, raw_data.shape[0], prefix='Progress:', suffix='Complete', length=15)

        sentences = []
        for i in range(raw_data.shape[0]):
            post = re.sub(r"[()]", "", raw_data.at[i, 'cleaned_post_text'])
            curr_sentences = post.split(". ")
            curr_sentences = [s for s in curr_sentences if s]
            for s in curr_sentences:
                doc = nlp(s)
                word_lst = [token.lemma_.lower() for token in doc if not token.is_space]
                sentences.append(word_lst)
            
            printProgressBar(i + 1, raw_data.shape[0], prefix='Progress:', suffix='Complete', length=15)
            
        with open('./datasets/processed/sentences.json', 'w') as fp:
            json.dump(sentences, fp)
        
        print("Finish Parsing")
    
    if not path.exists('word2vec.model'):
        print("Start Training Word2Vec Model")

        with open('./datasets/processed/sentences.json', 'rb') as fp:
            sentences = json.load(fp)
                
        model = Word2Vec(sentences, min_count=1) # sg = 1
        model.train(sentences, total_examples=len(sentences), epochs=95)
        model.save('word2vec.model')

        print("Finish Training")

    if not path.exists('./datasets/processed/embedding.json'):
        print("Start Building Dictionary for Word2Vec")

        with open('./datasets/processed/vocab.json', 'rb') as fp:
            vocab = json.load(fp)
        
        # printProgressBar(0, len(vocab), prefix='Progress:', suffix='Complete', length=15)

        model = Word2Vec.load('word2vec.model')
        data = dict()
        idx = 0
        for word in vocab:
            idx += 1
            printProgressBar(idx, len(vocab), prefix='Progress:', suffix='Complete', length=15)
            if word in model.wv.vocab:
                data[word] = model.wv[word].tolist()

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