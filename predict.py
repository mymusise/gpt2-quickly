from transformers import TextGenerationPipeline
from transformers import GPT2Tokenizer
from train import init_model, load_tokenizer

tokenizer = load_tokenizer()
model = init_model(tokenizer)

text_generator = TextGenerationPipeline(model, tokenizer)
seed = ['這是我第一次嘗試寫魯迅','我真心參加了閱讀心得比賽','我的離島記憶。','讀完一本書後，把自己的感想跟心得打出來','獨享，獨自享受，不願和其他人分享。']
print(seed)
for j in range(3):
  input=seed[j]
  for i in range(1):
    glist = text_generator(input, max_length=100, do_sample=True, top_k=10, repetition_penalty=2.0)
    gtext = glist[0]["generated_text"]
    if input in gtext:
      gtext = gtext.replace(input,'')
    print(gtext)
    input = gtext[-20:]
