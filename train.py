import tensorflow as tf
from transformers import GPT2Config, TFGPT2LMHeadModel
from transformers import TFGPT2LMHeadModel
from transformers import XLNetTokenizer
import configs
from official import nlp
import official.nlp.optimization
import click
import time
import pickle
from pathlib import Path
import numpy as np
# from tensorflow.keras.mixed_precision import experimental as mixed_precision


# policy = mixed_precision.Policy('mixed_float16')
# mixed_precision.set_policy(policy)


def load_tokenizer() -> XLNetTokenizer:
    tokenizer = XLNetTokenizer.from_pretrained(
        configs.data.path, max_len=configs.model.max_length, add_special_token=False)
    tokenizer.return_attention_mask = None
    return tokenizer


def get_dataset() -> tf.data.Dataset:
    p = Path(configs.data.path)
    pickle_files = p.glob('*.pickle')
    ids, labels = [], []
    for pickle_file in pickle_files:
        print(f"loading {pickle_file}")
        _ids, _labels = pickle.load(open(pickle_file, 'rb'))
        if len(ids) == 0:
            ids = _ids
            labels = _labels
        else:
            ids = np.vstack((ids, _ids))
            labels = np.vstack((labels, _labels))
    print(ids.shape, labels.shape, ids.dtype, labels.dtype)
    dataset = tf.data.Dataset.from_tensor_slices((
        ids,
        labels
    )).repeat().shuffle(ids.shape[0], reshuffle_each_iteration=True).batch(configs.model.batch_size)
    return dataset


def init_model(
    tokenizer: XLNetTokenizer,
    train_steps: int = 20000,
    num_warmup_steps: int = 1000,
    model_path: str = configs.model_path,
) -> TFGPT2LMHeadModel:

    try:
        model = TFGPT2LMHeadModel.from_pretrained(
            model_path, return_dict=False)
    except EnvironmentError:
        config = GPT2Config(
            architectures=["TFGPT2LMHeadModel"],
            model_type="TFGPT2LMHeadModel",
            tokenizer_class="XLNetTokenizer",
            vocab_size=tokenizer.vocab_size,
            n_positions=configs.model.n_positions,
            n_ctx=configs.model.n_ctx,
            n_embd=configs.model.n_embd,
            n_layer=configs.model.n_layer,
            n_head=configs.model.n_head,
            pad_token_id=tokenizer.pad_token_id,
            task_specific_params={
                "text-generation": {
                    "do_sample": True,
                    "max_length": 120
                }
            },
            return_dict=False,
            output_attentions=False,
            output_hidden_states=False,
            use_cache=False,
        )
        model = TFGPT2LMHeadModel(config)

    loss = model.compute_loss
    optimizer = nlp.optimization.create_optimizer(
        5e-6, num_train_steps=train_steps, num_warmup_steps=num_warmup_steps)

    metric = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')
    # metric = Mymetrice('accuracy')

    model.compile(
        optimizer=optimizer,
        loss=[loss, *[None] * model.config.n_layer],
        metrics=[metric]
    )

    return model


def train(model, train_dataset, epochs, train_steps):

    def scheduler(epoch, lr):
        return lr

    class AutoSaveCallback(tf.keras.callbacks.Callback):
        def on_train_batch_begin(self, batch, logs):
            save_pre_batch = 2000
            backup_num = 10
            if batch % save_pre_batch == 0:
                batch_no = batch // save_pre_batch % backup_num
                self.model.save_pretrained(f'{configs.model_path}{batch_no}')

        def on_epoch_end(self, epoch, logs=None):
            self.model.save_pretrained(f'{configs.model_path}')

    callbacks = [
        tf.keras.callbacks.TensorBoard(
            log_dir=f'{configs.model_path}/logs', update_freq=50),
        AutoSaveCallback()
    ]

    t1 = time.time()

    model.fit(
        train_dataset,
        epochs=epochs,
        steps_per_epoch=train_steps,
        callbacks=callbacks,
        batch_size=None
    )
    print(f'total train time {time.time() - t1}')


@click.command()
@click.option('--epochs', default=50, help='number of epochs')
@click.option('--train_steps', default=1000, help='number of train_steps')
def main(epochs, train_steps):
    warmup_steps = int(train_steps * epochs * 0.1)

    tokenizer = load_tokenizer()
    train_dataset = get_dataset()
    model = init_model(tokenizer, train_steps * epochs,
                       warmup_steps, configs.model_path)
    train(model, train_dataset, epochs, train_steps)


if __name__ == '__main__':
    mirrored_strategy = tf.distribute.MirroredStrategy()
    with mirrored_strategy.scope():
        main()
