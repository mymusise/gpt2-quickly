from train import load_tokenizer
from multiprocessing import Process, Manager, Queue
from tqdm import tqdm
from typing import List
import tensorflow as tf
import pickle
import json
import configs
import numpy as np
import click
import re


def encode_processer(processer_num, texts, result_dict):
    tokenizer = load_tokenizer()
    input_ids = []
    for text in tqdm(texts):
        text_encoded = tokenizer(
            text, return_attention_mask=False, return_token_type_ids=False, padding="max_length")['input_ids']
        input_ids += text_encoded
    print(f"assert {processer_num} with {len(input_ids)}")
    # result_dict[processer_num] = input_ids  

    block_size = configs.model.max_length
    text_len = len(input_ids) // block_size
    input_ids = np.array(input_ids)
    input_ids.resize(text_len * block_size)
    input_ids = input_ids.reshape((text_len, block_size))

    ids = input_ids[:, :-1]
    labels = input_ids[:, 1:]
    with open(configs.data.pickle.replace('.pickle', f'_{processer_num}.pickle'), 'wb') as f:
        pickle.dump((ids, labels), f)


def multiply_encode(handler, tasks):
    manager = Manager()
    result_dict = manager.dict()  # didn't work and don't know why
    jobs = []
    for processer_num, task in enumerate(tasks):
        p = Process(target=handler, args=(
            processer_num, task, result_dict))
        jobs.append(p)

    for job in jobs:
        job.start()

    for job in jobs:
        job.join()

    # input_ids = []  # don't know why it didn't work
    # for processer_num, ids in result_dict.items():
    #     input_ids += ids

    for job in jobs:
        try:
            job.close() # It may raise exception in python <=3.6
        except:
            pass
    print("[all_task done]")


def split_data(
        text,
        n_processes,
        block_size,
        split_token_re=r"(。|？|！|\n)",
) -> List[str]:
    texts = []
    datas = re.split(split_token_re, text)
    max_length = block_size - 2
    stack = ""
    print('merging data')
    for i, data in enumerate(datas):
        if len(stack) + len(data) <= max_length:
            stack += data
        else:
            texts.append(stack)
            if i > 0:
                if len(datas[i-1]) == 1 and i > 1:
                    pre_text = datas[i-2] + datas[i-1]
                else:
                    pre_text = datas[i-1]
                stack = pre_text + data
            else:
                stack = data
    texts.append(stack)

    print('merged')
    text_task = []
    num_pre_task = len(texts) // n_processes
    for i in range(0, len(texts), num_pre_task):
        text_task.append(texts[i: i + num_pre_task])
    return text_task


@click.command()
@click.option('--n_processes', default=1, help='Number of processes.')
def preprocess(n_processes):
    block_size = configs.model.max_length

    print(f'reading {configs.data.raw}')
    with open(configs.data.raw, 'r') as f:
        data = f.read().replace('  ', ' ').replace('\n\n', '\n')
        print(f"total words: {len(data)}")

    text_task = split_data(data, n_processes, block_size)
    print("num of task: ", len(text_task))

    multiply_encode(encode_processer, text_task)


if __name__ == '__main__':
    preprocess()
