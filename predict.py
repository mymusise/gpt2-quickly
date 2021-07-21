from transformers import TextGenerationPipeline
from transformers import GPT2Tokenizer
from train import init_model, load_tokenizer

tokenizer = load_tokenizer()
model = init_model(tokenizer)

text_generator = TextGenerationPipeline(model, tokenizer)
seed = '國文的世界真美妙'


for i in range(20):
  glist = text_generator(seed, seed, max_length=100, do_sample=True, top_k=30, repetition_penalty=2.0)
  gtext = glist[0]["generated_text"]
  if seed in gtext:
    gtext = gtext.replace(seed,'')
  print(gtext)
  seed = gtext[-20:]
