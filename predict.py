from transformers import TextGenerationPipeline
from transformers import GPT2Tokenizer
from train import init_model, load_tokenizer

tokenizer = load_tokenizer()
model = init_model(tokenizer)

text_generator = TextGenerationPipeline(model, tokenizer)
print(text_generator("大", max_length=8, do_sample=False))
print(text_generator("无尽", max_length=8, do_sample=False))
print(text_generator("大千世界", max_length=8, do_sample=False))
print(text_generator("第一章", max_length=8, do_sample=False))
