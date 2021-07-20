from transformers import TextGenerationPipeline
from transformers import GPT2Tokenizer
from train import init_model, load_tokenizer

tokenizer = load_tokenizer()
model = init_model(tokenizer)

text_generator = TextGenerationPipeline(model, tokenizer)
print(text_generator("國文的世界真美妙", max_length=100, do_sample=True, top_k=10, eos_token_id=tokenizer.get_vocab().get("】", 0)))
print(text_generator("國文的世界真美妙", max_length=100, do_sample=True, top_k=20, eos_token_id=tokenizer.get_vocab().get("】", 0)))
print(text_generator("國文的世界真美妙", max_length=100, do_sample=True, top_k=30, eos_token_id=tokenizer.get_vocab().get("】", 0)))
print(text_generator("國文的世界真美妙", max_length=100, do_sample=False))
print(text_generator("國文的世界真美妙", max_length=100, do_sample=False))
