from transformers import TextGenerationPipeline
from transformers import GPT2Tokenizer
from train import init_model, load_tokenizer

tokenizer = load_tokenizer()
model = init_model(tokenizer)

text_generator = TextGenerationPipeline(model, tokenizer)
seed = ['這是我第一次嘗試寫','生活中充滿著大大小小的','108上學期小王子電子書','起初我聽朋友介紹','本作品參加108學年度','參加中學生閱讀心得']
print(seed)
for j in range(5):
  input=seed[j]
  for i in range(5):
    glist = text_generator(input, max_length=100, do_sample=True, top_k=10, repetition_penalty=2.0)
    gtext = glist[0]["generated_text"]
    if input in gtext:
      gtext = gtext.replace(input,'')
    print(gtext)
    input = gtext[-20:]
