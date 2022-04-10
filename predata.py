from train import load_tokenizer
from multiprocessing import Process
from tqdm import tqdm
from typing import List
import tensorflow as tf
import pickle
import json
import configs
import numpy as np
import click
import os
import random
import time
import gc


def encode_processer(processer_num: int):
    tokenizer = load_tokenizer()
    contents = pickle.load(open(os.path.join(configs.data.pickle_path, f'data.{processer_num}.jsonp'), 'rb'))
    contents = contents.split('\n\n|-|\n\n')
    output_file_l = open(os.path.join(configs.data.pickle_path, f'data.{processer_num}.l.pickle'), 'wb')
    output_file_m = open(os.path.join(configs.data.pickle_path, f'data.{processer_num}.m.pickle'), 'wb')
    output_file_s = open(os.path.join(configs.data.pickle_path, f'data.{processer_num}.s.pickle'), 'wb')
    for content in tqdm(contents, desc=f"processer_{processer_num}"):
        if len(content) < 24:
            continue
        if len(content) <= 64 - 2:
            pre_size = 64
            output_file = output_file_s
        elif 64 - 2 < len(content) <= 128 - 2:
            pre_size = 128
            output_file = output_file_m
        else:
            pre_size = configs.model.max_length 
            output_file = output_file_l
        content = tokenizer.sep_token + content + tokenizer.cls_token
        content_decoded = tokenizer(content, return_attention_mask=False,
                                    return_token_type_ids=False, add_special_tokens=False)['input_ids']

        if len(content_decoded) > pre_size:
            end_left_size = 64
            block_size = (pre_size - end_left_size)
            block_num = (len(content_decoded) - pre_size) // block_size
            block_num += 1 if (len(content_decoded) -
                            pre_size) % block_size != 0 else 0
            new_content = [content_decoded[:pre_size]]
            for i in range(block_num):
                _block = content_decoded[pre_size + i *
                                        block_size-end_left_size:pre_size + (i+1)*block_size]
                new_content.append(_block)
        else:
            new_content = [content_decoded]
        if len(new_content[-1]) < pre_size and len(new_content) > 1:
            new_content[-1] = new_content[-2][-(pre_size - len(new_content[-1])):] + new_content[-1]
        else:
            new_content[-1] = new_content[-1] + [tokenizer.pad_token_id] * (pre_size - len(new_content[-1]))
        if len(new_content) > 0:
            input_ids = np.array(new_content, dtype=np.int32)
            output_file.write(pickle.dumps(input_ids)+'换行'.encode())



def multiply_encode(handler, n_processes):
    jobs = []
    for processer_num in range(n_processes):
        p = Process(target=handler, args=(
            processer_num, ))
        jobs.append(p)

    for job in jobs:
        job.start()

    for job in jobs:
        job.join()

    for job in jobs:
        job.close()  # It may raise exception in python <=3.6
    print("[all_task done]")


def split_data(
        texts,
        n_processes,
        block_size,
        split_token_re=r"(。|？|！|\n)",
) -> List[str]:
    num_pre_task = len(texts) // n_processes
    print(num_pre_task, len(texts), n_processes)
    for _i, i in tqdm(enumerate(range(0, len(texts), num_pre_task)), desc="spliting data..."):
        current_text = texts[i: i + num_pre_task]
        with open(os.path.join(configs.data.pickle_path, f'data.{_i}.jsonp'), 'wb') as output_file:
            pickle.dump(current_text, output_file)
        print("pre task num: ", len(current_text))
        del current_text


@click.command()
@click.option('--n_processes', default=1, help='Number of processes.')
def preprocess(n_processes):
    block_size = configs.model.max_length

    print(f'reading {configs.data.raw}')

    data = []
    print('reading raw data ...')
    with open(configs.data.raw, 'r') as f:
        for line in tqdm(list(f.readlines())):
            if len(line) > 0:
                data.append(line)
            del line
    random.shuffle(data)
    data = "\n\n|-|\n\n".join(data)
    split_data(data, n_processes, block_size)
    del data
    gc.collect()
    # time.sleep(1000)
    multiply_encode(encode_processer, n_processes)


if __name__ == '__main__':
    preprocess()
