from transformers import BertTokenizer
from train import load_tokenizer
from multiprocessing import Process, Manager
from tqdm import tqdm
import tensorflow as tf
import pickle
import configs
import numpy as np
import click


def encode_processer(processer_num, texts, result_dict):
    tokenizer = load_tokenizer()
    input_ids = []
    for text in tqdm(texts):
        text_encoded = tokenizer(text)['input_ids']
        input_ids += text_encoded
    result_dict[processer_num] = input_ids


def multiply_encode(handler, tasks):
    manager = Manager()
    result_dict = manager.dict()
    jobs = []
    for processer_num in range(len(tasks)):
        p = Process(target=handler, args=(
            processer_num, tasks[processer_num], result_dict))
        jobs.append(p)
        p.start()

    for job in jobs:
        job.join()
    print("[all_task done]")
    return result_dict


def split_data(data, n_processes, block_size):
    texts = []
    for i in range(0, len(data), block_size):
        text = data[i:i+block_size]
        texts.append(text)

    text_task = []
    num_pre_task = len(texts) // n_processes
    for i in range(0, len(texts), num_pre_task):
        text_task.append(texts[i: i + num_pre_task])
    return text_task


@click.command()
@click.option('--n_processes', default=2, help='Number of processes.')
def preprocess(n_processes):
    block_size = configs.model.max_length

    input_ids = []
    with open(configs.data.raw, 'r') as f:
        data = f.read().replace('  ', ' ').replace('\n\n', '\n')
        print(f"total words: {len(data)}")

    text_task = split_data(data, n_processes, block_size)
    print("len(text_task)", len(text_task))

    result_dict = multiply_encode(encode_processer, text_task)
    for processer_num, ids in result_dict.items():
        input_ids += ids

    text_len = len(input_ids) // block_size
    input_ids = np.array(input_ids)
    input_ids.resize(text_len * block_size)
    input_ids = input_ids.reshape((text_len, block_size))

    ids = input_ids[:, :-1]
    labels = input_ids[:, 1:]
    print(f"ids.shape: {ids.shape}  labels.shape: {labels.shape}")
    print("dumping...")
    pickle.dump((ids, labels), open(configs.data.pickle, 'wb'))


if __name__ == '__main__':
    preprocess()
