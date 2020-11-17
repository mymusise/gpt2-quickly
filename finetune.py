
from transformers import BertTokenizer, TFGPT2LMHeadModel
from transformers import GPT2Config, TFGPT2LMHeadModel
from transformers import TextGenerationPipeline
from official import nlp
import official.nlp.optimization
from train import load_tokenizer, train, get_dataset
import tensorflow as tf
from configs import finetune as configs
import click


def load_model(train_steps, num_warmup_steps):
    try:  # try to load finetuned model at local.
        tokenizer = load_tokenizer()
        config = GPT2Config.from_pretrained(configs.model_path, return_dict=False)
        model = TFGPT2LMHeadModel.from_pretrained(configs.model_path, return_dict=False)
        print("model loaded from local!")
    except Exception as e:
        tokenizer = BertTokenizer.from_pretrained(
            "mymusise/gpt2-medium-chinese")
        model = TFGPT2LMHeadModel.from_pretrained(
            "mymusise/gpt2-medium-chinese", return_dict=False)
        print("model loaded from remote!")

    loss = model.compute_loss
    optimizer = nlp.optimization.create_optimizer(
        5e-5, num_train_steps=train_steps, num_warmup_steps=num_warmup_steps)
    metric = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')

    model.compile(
        optimizer=optimizer,
        loss=[loss, *[None] * model.config.n_layer],
        # metrics=[metric]
    )
    return model


@click.command()
@click.option('--epochs', default=20, help='number of epochs')
@click.option('--train_steps', default=2000, help='number of train_steps')
def finetune(epochs, train_steps):
    warmup_steps = int(train_steps * epochs * 0.1)

    train_dataset = get_dataset()
    model = load_model(train_steps, warmup_steps)
    train(model, train_dataset, epochs, train_steps)


if __name__ == '__main__':
    finetune()
