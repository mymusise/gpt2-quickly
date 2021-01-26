from tokenizers.trainers import WordPieceTrainer
from tokenizers import Tokenizer
from tokenizers.models import BPE
import configs
import os


tokenizer = Tokenizer(BPE())
trainer = WordPieceTrainer(special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])

def main():
    tokenizer.train(trainer, [configs.data.raw])
    tokenizer.save(os.path.join(configs.data.path, 'bpe.vocab'))
    print(f"save to {configs.data.path}")


def train_with_sentenceprices(vocab_size: int = 3000, num_threads=2, character_coverage=0.98):
    os.system(f"spm_train --input={configs.data.raw} --model_prefix=spiece --model_type=bpe --character_coverage={character_coverage} --vocab_size={vocab_size} --num_threads={num_threads}")
    os.system(f"mv spiece.model {configs.data.path}")


if __name__ == '__main__':
    train_with_sentenceprices()
