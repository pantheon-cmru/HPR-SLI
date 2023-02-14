#import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
model_name = 't5_gec_model'
torch_device = 'cpu' #'cuda' if torch.cuda.is_available() else
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name).to(torch_device)

def correct_grammar(input_text,num_return_sequences):
  batch = tokenizer([input_text],truncation=True,padding='max_length',max_length=64, return_tensors="pt").to(torch_device)
  translated = model.generate(**batch,max_length=64,num_beams=10, num_return_sequences=num_return_sequences, temperature=1.5)
  tgt_text = tokenizer.batch_decode(translated, skip_special_tokens=True)
  return tgt_text



text = input()
print(correct_grammar(text, num_return_sequences=1))

'''text1 = 'Me is Rashmi'
print(correct_grammar(text1, num_return_sequences=1))

text2 = 'I like play games'
print(correct_grammar(text2, num_return_sequences=1))'''