from transformers import TFGPT2Model, GPT2Config, TFGPT2LMHeadModel
from transformers import TFTrainer, TFTrainingArguments
from transformers import TFGPT2LMHeadModel, tf_top_k_top_p_filtering
from transformers import GPT2TokenizerFast, AutoTokenizer, BertTokenizer, BertTokenizerFast
from transformers import GPT2Config
import tensorflow as tf
import json
import numpy as np


max_length = 1023


tokenizer = BertTokenizer.from_pretrained(
    '/data/novels/vocab-bert/', max_len=max_length)
data_encoding = tokenizer(
    ["你好", "今天"], is_split_into_words=True,  padding=True, truncation=True)
print(data_encoding)


datas = []
labels = []
with open('/home/mymusise/pro/gpt2-ml/dataset/train_data/dazhuzai.mini.json', 'r') as f:
    for l_no, l_text in enumerate(f):
        text = json.loads(l_text)['text']
        datas.append(text)

text_encoded = tokenizer(datas)
for k in text_encoded.keys():
    if k == 'input_ids':
        fill_id = tokenizer.pad_token_id
    elif k == 'token_type_ids':
        fill_id = 0
    elif k == 'attention_mask':
        fill_id = 1
    else:
        fill_id = 0

    for i, ids in enumerate(text_encoded[k]):
        text_encoded[k][i] = ids + [fill_id] * \
            (max_length - len(text_encoded[k][i]))

for ids in text_encoded['input_ids']:
    labels.append(ids[1:] + [tokenizer.pad_token_id])

input_data = {}
for k in text_encoded.keys():
    input_data[k] = np.array(text_encoded[k])
    # input_data[k] = text_encoded[k]
labels = np.array(labels)

train_dataset = tf.data.Dataset.from_tensor_slices((
    input_data,
    labels
)).repeat().shuffle(100)
print(len(labels), len(input_data['input_ids']), len(
    input_data['input_ids'][0]),  train_dataset)


config = GPT2Config(
    vocab_size=tokenizer.vocab_size,
    n_layer=4,
    n_head=4,
    # n_ctx=255,
)
model = TFGPT2LMHeadModel(config)

# training_args = TFTrainingArguments(
#     output_dir='/tmp/tf_train',          # output directory
#     num_train_epochs=3,              # total # of training epochs
#     per_device_train_batch_size=1,  # batch size per device during training
#     per_device_eval_batch_size=1,   # batch size for evaluation
#     warmup_steps=5,                # number of warmup steps for learning rate scheduler
#     weight_decay=0.01,           # strength of weight decay
#     logging_dir='./logs',            # directory for storing logs
#     max_steps=10,
#     save_steps=5,
#     logging_first_step=True,
#     logging_steps=2,
#     do_eval=False,
# )

# trainer = TFTrainer(
#     model=model,
#     args=training_args,
#     # data_collator=data_collator,
#     train_dataset=train_dataset,
#     prediction_loss_only=True,
# )

# trainer.train()


def cross_entropy_loss_with_padding(num_labels, pad_token_id):
    loss_fct = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction=tf.keras.losses.Reduction.NONE
    )

    def loss(y_true, y_pred):
        input_mask = tf.not_equal(y_true, pad_token_id)
        active_loss = tf.reshape(input_mask, (-1,))
        logits = tf.reshape(y_pred, (-1, num_labels))
        active_logits = tf.boolean_mask(logits, active_loss)

        train_labels = tf.reshape(y_true, (-1,))
        active_labels = tf.boolean_mask(train_labels, active_loss)
        cross_entropy = loss_fct(active_labels, active_logits)

        return cross_entropy

    return loss

optimizer = tf.keras.optimizers.Adam(learning_rate=3e-5)
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE)
# loss = cross_entropy_loss_with_padding(
#     num_labels=len(tokenizer), pad_token_id=tokenizer.pad_token_id,
# )

model.compile(optimizer=optimizer, loss=[loss, *[None] * model.config.n_layer])


model.fit(train_dataset, epochs=2, steps_per_epoch=100, batch_size=1)

model.save('./models/')
