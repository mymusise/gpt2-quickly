from numpy.core.shape_base import block
import tensorflow as tf
from transformers import BertTokenizer
from train import load_tokenizer
import pickle
import configs
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool


def add_token(texts, ids, tokenizer):
    return


def multiply_decode(handler, tasks, workers=None, call_back=None):
    if workers == None:
        workers = len(tasks)
    p = Pool(workers)
    for i in range(len(tasks)):
        p.apply_async(handler, tasks[i])
    p.close()
    p.join()
    print("[all_task done]")


def preprocess():
    block_size = configs.model.max_length
    tokenizer = load_tokenizer()
    input_ids = []
    with open(configs.data.raw, 'r') as f:
        data = f.read().replace('  ', ' ').replace('\n\n', '\n')
        print(0, len(data), block_size)
        texts = []
        for i in tqdm(range(0, len(data), block_size)):
            text = data[i:i+block_size]
            text_encoded = tokenizer(text)['input_ids']
            input_ids += text_encoded
    print(len(data), len(input_ids))
    text_len = len(input_ids) // block_size
    input_ids = np.array(input_ids)
    input_ids.resize(text_len * block_size)
    input_ids = input_ids.reshape((text_len, block_size))

    ids = input_ids[:, :-1]
    labels = input_ids[:, 1:]
    print(f"ids.shape: {ids.shape}  labels.shape: {labels.shape}")
    pickle.dump((ids, labels), open(configs.data.pickle, 'wb'))


if __name__ == '__main__':
    preprocess()
