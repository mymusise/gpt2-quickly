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
    max_length = configs.model.max_length
    tokenizer = load_tokenizer()
    input_ids = []
    size_left = 32  # TODO: make size_left dynamical
    last_encoded = None
    real_size = max_length - size_left

    for i in tqdm(range(len(texts) // 50)):
        text = "".join(texts[i * 50: (i + 1) * 50])
        text = text.replace('\n\n', '\n')
        text_encoded = tokenizer(text, return_attention_mask=False,
                                    return_token_type_ids=False, add_special_tokens=False)['input_ids']  # list
        if last_encoded is not None:
            text_encoded = last_encoded + text_encoded
        batch_size = len(text_encoded) // real_size
        last_size = len(text_encoded) % real_size
        for i in range(batch_size):
            if i == 0:
                # [0, 0, ...] 1*size_left
                fill_encoded = np.zeros([size_left], dtype=np.int).tolist()
            else:
                # last <size_left> token
                fill_encoded = text_encoded[real_size*i-size_left:real_size*i]
            current_encoded = fill_encoded + text_encoded[real_size*i: real_size*(i+1)]

            assert len(current_encoded) == max_length

            input_ids.append(current_encoded)

        if last_size != 0:
            last_encoded = text_encoded[-last_size:]
        else:
            last_encoded = None

    print(f"assert {processer_num} with {len(input_ids)}")

    input_ids = np.array(input_ids)
    input_ids = input_ids[1:]
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
            stack = ""
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
