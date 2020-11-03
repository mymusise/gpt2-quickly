from transformers import TFGPT2Model, GPT2Config, TFGPT2LMHeadModel
from transformers import TFTrainer, TFTrainingArguments
from transformers import TFGPT2LMHeadModel
from transformers import BertTokenizer, BertTokenizerFast, GPT2Tokenizer
from transformers import GPT2Config
import tensorflow as tf
import tensorflow_addons as tfa
import json
from configs import test as configs
import numpy as np
from transformers import TextGenerationPipeline
import time


max_length = configs.configs['max_length']


def load_tokenizer() -> BertTokenizer:
    tokenizer = BertTokenizer.from_pretrained(
        configs.data['path'], max_len=max_length-1)
    return tokenizer


def get_dataset(
        path: str = configs.data['raw'],
        tokenizer: BertTokenizer = load_tokenizer()
) -> tf.data.Dataset:
    block_size = max_length
    with open(path, 'r') as f:
        data = f.read() * 1000
        tokenized_text = tokenizer(data, return_tensors='np')['input_ids']
        data_len = int(tokenized_text.shape[1] // block_size)
        print(len(data), tokenized_text.shape, data_len, block_size)
        input_ids = tokenized_text[:, :block_size *
                                   data_len].reshape((data_len, block_size))

    ids = input_ids[:, :-1]
    labels = input_ids[:, 1:]

    print(ids[:5])
    print(labels[:5])

    print(ids.shape, labels.shape, tokenizer.vocab_size)
    dataset = tf.data.Dataset.from_tensor_slices((
        ids,
        labels
    )).batch(12)
    return dataset


def init_model(tokenizer):

    try:
        model = TFGPT2LMHeadModel.from_pretrained(
            './models/199', return_dict=True)
    except EnvironmentError:
        config = GPT2Config(
            vocab_size=tokenizer.vocab_size,
            n_positions=512,
            n_ctx=512,
            n_embd=768,
            n_layer=8,
            n_head=12,
            pad_token_id=tokenizer.pad_token_id,
        )
        model = TFGPT2LMHeadModel(config)

    # optimizer = tfa.optimizers.RectifiedAdam(
    #     warmup_proportion=0.1,
    #     min_lr=1e-8,
    # )
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    # loss = cross_entropy_loss_with_padding(
    #     num_labels=tokenizer.vocab_size,
    #     pad_token_id=tokenizer.pad_token_id
    # )
    optimizer = tf.keras.optimizers.Adam(learning_rate=3e-6, epsilon=1e-08)
    metric = tf.keras.metrics.SparseCategoricalCrossentropy(
        name='accuracy', from_logits=True)
    metrics = [
        tf.keras.metrics.SparseCategoricalAccuracy('accuracy')
    ]

    model.compile(
        optimizer=optimizer,
        loss=[loss, *[None] * model.config.n_layer],
        # metrics=metrics,
    )

    return model


def train():
    tokenizer = load_tokenizer()

    class AutoSaveCallback(tf.keras.callbacks.Callback):
        def on_train_batch_begin(self, batch, logs=None):
            if (batch + 1) % 200 == 0:
                self.model.save_pretrained(f'./models/{batch}')

        def on_epoch_end(self, epoch, logs=None):
            text_generator = TextGenerationPipeline(self.model, tokenizer)
            print(text_generator("大", max_length=5, do_sample=False))
            print(text_generator("大千", max_length=max_length-1, do_sample=False))

    callbacks = [
        tf.keras.callbacks.TensorBoard(log_dir='./logs'),
        tf.keras.callbacks.ModelCheckpoint(filepath='./models',
                                           save_weights_only=True),
        AutoSaveCallback()
    ]

    train_dataset = get_dataset(tokenizer=tokenizer)

    # with tf.device("gpu:0"):
    model = init_model(tokenizer)

    model.fit(train_dataset, epochs=10, callbacks=callbacks, batch_size=12)

    model.save_pretrained('./models/')


if __name__ == '__main__':
    with tf.device('/gpu:1'):
        train()
