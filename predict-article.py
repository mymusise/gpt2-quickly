from transformers import TextGenerationPipeline
from transformers import GPT2Tokenizer
from train import init_model, load_tokenizer
import string
import random
import datetime
from datetime import datetime

now = datetime.now()

tokenizer = load_tokenizer()
model = init_model(tokenizer)

text_generator = TextGenerationPipeline(model, tokenizer)

path = '/content/drive/MyDrive/100word/test/raw.txt'

pathout = '/content/drive/MyDrive/100word/test/國文課程成果-'+ now.strftime("%d-%m-%Y-%H-%M-%S") +'.txt'

f = open(path, 'r',encoding="utf-8")
fout = open(pathout, 'w',encoding="utf-8")

dtitle = '國文課程學習成果100字簡述- '+ now.strftime("%d/%m/%Y %H:%M:%S") + '\n\n'
fout.write(dtitle)

for k in range(10):
  rand = random.randint(1,100)
  for r in range(rand):
    line = f.readline()

  original_line = line.strip('\n')
  print(k,'原 : ',original_line)
  fout.write(original_line)

#產出一段一段
  for d in range(10):
    feedin = (original_line[:10])

    glist = text_generator(feedin, max_length=250, do_sample=True, top_k=20, repetition_penalty=1.3)
    output_sentence = str(k).zfill(2) + str(d) + '段: '+ glist[0]["generated_text"] + '\n' 
    print(output_sentence)

    if feedin in output_sentence:
      output_sentence = output_sentence.replace(feedin,'')

    fout.write(output_sentence)
    feedin = output_sentence[230:250]


f.close()
fout.close()


#seed = ['魯迅是一個好作者','化學實驗真是好玩','我的物理考試很笨。','讀完一本書後，把自己的感想跟心得打出來','獨享，獨自享受，不願和其他人分享。']
#print(seed)
#for j in range(1):
#  input=seed[j]
#  print('段落',j)
#  for i in range(1):
#    glist = text_generator(input, max_length=95, do_sample=True, top_k=10, repetition_penalty=2.0)
#    gtext = glist[0]["generated_text"]
#    if input in gtext:
#      gtext = gtext.replace(input,'')
#    print(gtext)
#    input = gtext[-20:]
