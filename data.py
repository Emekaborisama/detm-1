import os
import random
import pickle
import numpy as np
import pandas as pd
import torch 
import scipy.io
import json

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def _fetch(path, name):
    if name == 'train':
        token_file = os.path.join(path, 'bow_tr_tokens.mat')
        count_file = os.path.join(path, 'bow_tr_counts.mat')
    elif name == 'valid':
        token_file = os.path.join(path, 'bow_va_tokens.mat')
        count_file = os.path.join(path, 'bow_va_counts.mat')
    else:
        token_file = os.path.join(path, 'bow_ts_tokens.mat')
        count_file = os.path.join(path, 'bow_ts_counts.mat')
    tokens = scipy.io.loadmat(token_file)['tokens'].squeeze()
    counts = scipy.io.loadmat(count_file)['counts'].squeeze()
    if name == 'test':
        token_1_file = os.path.join(path, 'bow_ts_h1_tokens.mat')
        count_1_file = os.path.join(path, 'bow_ts_h1_counts.mat')
        token_2_file = os.path.join(path, 'bow_ts_h2_tokens.mat')
        count_2_file = os.path.join(path, 'bow_ts_h2_counts.mat')
        tokens_1 = scipy.io.loadmat(token_1_file)['tokens'].squeeze()
        counts_1 = scipy.io.loadmat(count_1_file)['counts'].squeeze()
        tokens_2 = scipy.io.loadmat(token_2_file)['tokens'].squeeze()
        counts_2 = scipy.io.loadmat(count_2_file)['counts'].squeeze()
        return {'tokens': tokens, 'counts': counts, 'tokens_1': tokens_1, 'counts_1': counts_1, 'tokens_2': tokens_2, 'counts_2': counts_2}
    return {'tokens': tokens, 'counts': counts}

def _fetch_temporal(path, name):
    if name == 'train':
        token_file = os.path.join(path, 'bow_tr_tokens.mat')
        count_file = os.path.join(path, 'bow_tr_counts.mat')
        time_file = os.path.join(path, 'bow_tr_timestamps.mat')
    elif name == 'valid':
        token_file = os.path.join(path, 'bow_va_tokens.mat')
        count_file = os.path.join(path, 'bow_va_counts.mat')
        time_file = os.path.join(path, 'bow_va_timestamps.mat')
    else:
        token_file = os.path.join(path, 'bow_ts_tokens.mat')
        count_file = os.path.join(path, 'bow_ts_counts.mat')
        time_file = os.path.join(path, 'bow_ts_timestamps.mat')
    tokens = scipy.io.loadmat(token_file)['tokens'].squeeze()
    counts = scipy.io.loadmat(count_file)['counts'].squeeze()
    times = scipy.io.loadmat(time_file)['timestamps'].squeeze()
    if name == 'test':
        token_1_file = os.path.join(path, 'bow_ts_h1_tokens.mat')
        count_1_file = os.path.join(path, 'bow_ts_h1_counts.mat')
        token_2_file = os.path.join(path, 'bow_ts_h2_tokens.mat')
        count_2_file = os.path.join(path, 'bow_ts_h2_counts.mat')
        tokens_1 = scipy.io.loadmat(token_1_file)['tokens'].squeeze()
        counts_1 = scipy.io.loadmat(count_1_file)['counts'].squeeze()
        tokens_2 = scipy.io.loadmat(token_2_file)['tokens'].squeeze()
        counts_2 = scipy.io.loadmat(count_2_file)['counts'].squeeze()
        return {'tokens': tokens, 'counts': counts, 'times': times, 
                    'tokens_1': tokens_1, 'counts_1': counts_1, 
                        'tokens_2': tokens_2, 'counts_2': counts_2} 
    return {'tokens': tokens, 'counts': counts, 'times': times}

def _fetch_json_temporal(path):
    data = pd.read_json("./datasets/processed/vaccine-forums-processed.json")
    tokens = data['tokens'].tolist()
    counts = data['counts'].tolist()
    times = data['post_updated_at'].tolist()

    return {'tokens': tokens, 'counts': counts, 'times': times}

_fetch_json_temporal("")

def get_data(path, temporal=False):
    ### load vocabulary
    with open(os.path.join(path, 'vocab.pkl'), 'rb') as f:
        vocab = pickle.load(f)

    if not temporal:
        train = _fetch(path, 'train')
        valid = _fetch(path, 'valid')
        test = _fetch(path, 'test')
    else:
        train = _fetch_temporal(path, 'train')
        valid = _fetch_temporal(path, 'valid')
        test = _fetch_temporal(path, 'test')

    return vocab, train, valid, test

def get_json_data(path):
    ### load vocabulary
    with open(os.path.join(path, 'vocab.json'), 'rb') as f:
        vocab = json.load(f)
    
    data = _fetch_json_temporal(path)
    ### 70% training 20% validation 10% test
    card = len(data['tokens'])
    training_num = int(card * 0.7)
    validation_num = int(card * 0.2)

    train = dict()
    valid = dict()
    test = dict()

    train['tokens'] = data['tokens'][:training_num]
    train['counts'] = data['counts'][:training_num]
    train['times'] = data['times'][:training_num]

    valid['tokens'] = data['tokens'][training_num : training_num + validation_num]
    valid['counts'] = data['counts'][training_num : training_num + validation_num]
    valid['times'] = data['times'][training_num : training_num + validation_num]

    test['tokens'] = data['tokens'][training_num + validation_num :]
    test['counts'] = data['counts'][training_num + validation_num :]
    test['times'] = data['times'][training_num + validation_num :]

    test_num_1 = int(card * 0.05)
    test['tokens_1'] = data['tokens'][training_num + validation_num : training_num + validation_num + test_num_1]
    test['counts_1'] = data['counts'][training_num + validation_num : training_num + validation_num + test_num_1]
    test['tokens_2'] = data['tokens'][training_num + validation_num + test_num_1 :]
    test['counts_2'] = data['counts'][training_num + validation_num + test_num_1 :]
        
    return vocab, train, valid, test

# print(get_data("./datasets/processed/vaccine-forums-processed.json"))

def get_batch(tokens, counts, ind, vocab_size, emsize=100, temporal=False, times=None):
    """fetch input data by batch."""
    batch_size = len(ind)
    data_batch = np.zeros((batch_size, vocab_size))
    if temporal:
        times_batch = np.zeros((batch_size, ))
    for i, doc_id in enumerate(ind):
        doc = tokens[doc_id]
        count = counts[doc_id]
        if temporal:
            timestamp = times[doc_id]
            times_batch[i] = timestamp
        # if len(doc) == 1: 
        #     doc = [doc.squeeze()]
        #     count = [count.squeeze()]
        # else:
        #     doc = doc.squeeze()
        #     count = count.squeeze()
        doc = doc
        count = count
        if doc_id != -1:
            for j, word in enumerate(doc):
                data_batch[i, word] = count[j]
    data_batch = torch.from_numpy(data_batch).float().to(device)
    if temporal:
        times_batch = torch.from_numpy(times_batch).to(device)
        return data_batch, times_batch
    return data_batch

def get_rnn_input(tokens, counts, times, num_times, vocab_size, num_docs):
    indices = torch.randperm(num_docs)
    indices = torch.split(indices, 500) 
    rnn_input = torch.zeros(num_times, vocab_size).to(device)
    cnt = torch.zeros(num_times, ).to(device)
    for idx, ind in enumerate(indices):
        data_batch, times_batch = get_batch(tokens, counts, ind, vocab_size, temporal=True, times=times)
        for t in range(num_times):
            tmp = (times_batch == t).nonzero()
            docs = data_batch[tmp].squeeze().sum(0)
            rnn_input[t] += docs
            cnt[t] += len(tmp)
        if idx % 20 == 0:
            print('idx: {}/{}'.format(idx, len(indices)))
    rnn_input = rnn_input / cnt.unsqueeze(1)
    return rnn_input
