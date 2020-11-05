import tensorflow as tf
from transformers import BertTokenizer
from train import load_tokenizer
import pickle
import configs


def preprocess():
    block_size = configs.model.max_length
    tokenizer = load_tokenizer()
    with open(configs.data.raw, 'r') as f:
        data = f.read()
        tokenized_text = tokenizer(data, max_length=len(
            data), return_tensors='np')['input_ids']
        data_len = int(tokenized_text.shape[1] // block_size)
        print(len(data), tokenized_text.shape, data_len, block_size)
        input_ids = tokenized_text[:, :block_size * data_len]
        input_ids = input_ids.reshape((data_len, block_size))

    ids = input_ids[:, :-1]
    labels = input_ids[:, 1:]

    pickle.dump((ids, labels), open(configs.data.pickle, 'wb'))


if __name__ == '__main__':
    preprocess()
