from tokenizers import BertWordPieceTokenizer, CharBPETokenizer
from transformers import BertTokenizer, BertTokenizerFast, GPT2Tokenizer
import configs


def main():
    tokenizer = BertWordPieceTokenizer()
    tokenizer.train(files=[configs.data.raw],
                    vocab_size=52_000, min_frequency=5)
    tokenizer.save_model(configs.data.path)
    print(f"save to {configs.data.path}")
    # GPT2Tokenizer.from_pretrained()

if __name__ == '__main__':
    main()
