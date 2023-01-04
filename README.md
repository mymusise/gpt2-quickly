<h1 align="center">
GPT2 Quickly
</h1>

<h3 align="center">
<p>Build your own GPT2 quickly, without doing many useless work.
</h3>

<p align="center">
    <a href="https://colab.research.google.com/github/mymusise/gpt2-quickly/blob/main/examples/gpt2_quickly.ipynb">
        <img alt="Build" src="https://colab.research.google.com/assets/colab-badge.svg">
    </a>
</p>

This project is base on 🤗 transformer. This tutorial show you how to train your own language(such as Chinese or Japanese) GPT2 model in a few code with Tensorflow 2.

You can try this project in [colab](https://colab.research.google.com/github/mymusise/gpt2-quickly/blob/main/examples/gpt2_quickly.ipynb) right now.

## Main file

``` 

├── configs
│   ├── test.py
│   └── train.py
├── build_tokenizer.py
├── predata.py
├── predict.py
└── train.py
```

## Preparation


### virtualenv
``` bash
git clone git@github.com:mymusise/gpt2-quickly.git
cd gpt2-quickly
python3 -m venv venv
source venv/bin/activate

pip install -r requirements.txt
```

### Install google/sentencepiece

- see [https://github.com/google/sentencepiece#installation](https://github.com/google/sentencepiece#installation)


## 0x00. prepare your raw dataset

this is a example of raw dataset: [raw.txt](dataset/test/raw.txt)


## 0x01. Build vocab

```bash
python cut_words.py
python build_tokenizer.py
```


## 0x02. Tokenize

```bash
python predata.py --n_processes=2
```


## 0x03 Train

```bash
python train.py
```


## 0x04 Predict

```bash
python predict.py
```

## 0x05 Fine-Tune

```bash
ENV=FINETUNE python finetune.py
```
