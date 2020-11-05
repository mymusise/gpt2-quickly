<h1 align="center">
GPT2 Quickly
</h1>

<h3 align="center">
<p>Build your own GPT2 quickly, without doing many useless work.
</h3>

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mymusise/gpt2-quickly/blob/main/examples/gpt2_quickly.ipynb)

This project is base on 🤗 transformer. This tutorial show how to train your own GPT2 model in a few code with Tensorflow 2.

## Main file

``` 

├── build_tokenizer.py
├── configs
│   ├── __init__.py
│   ├── test.py
│   └── train.py
├── predata.py
├── predict.py
└── train.py
```

## Preparation

``` bash
git clone git@github.com:mymusise/gpt2-quickly.git
cd gpt2-quickly
python3 -m venv venv
source venv/bin/activate

pip install -r requirements.txt
```

## 0x00. prepare your raw dataset

this is a example of raw dataset: [raw.txt](dataset/test/raw.txt)


## 0x01. Build vocab

```bash
python build_tokenizer.py
```


## 0x02. Tokenize

```bash
python predata.py
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
```