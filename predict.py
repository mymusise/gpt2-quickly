from transformers import TextGenerationPipeline
from transformers import GPT2Tokenizer
from train import init_model, load_tokenizer

tokenizer = load_tokenizer()
model = init_model(tokenizer)

text_generator = TextGenerationPipeline(model, tokenizer)
print(text_generator("唐诗：", max_length=64, do_sample=True, top_k=10, eos_token_id=tokenizer.get_vocab().get("】", 0)))
print(text_generator("此地是我开", max_length=64, do_sample=True, top_k=10, eos_token_id=tokenizer.get_vocab().get("】", 0)))
print(text_generator("一只乌鸦", max_length=64, do_sample=True, top_k=10, eos_token_id=tokenizer.get_vocab().get("】", 0)))
print(text_generator("走向森林 ", max_length=64, do_sample=False))
print(text_generator("拿出一本秘籍", max_length=64, do_sample=False))
print(text_generator("今日", max_length=64, do_sample=False))
print(text_generator("大江东去", max_length=64, do_sample=False))
