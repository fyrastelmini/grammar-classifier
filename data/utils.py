import re
import random

def preprocess(sentence):

  lowercase_string = sentence.lower()

  processed_string = re.sub(r'[.,?!$%^:;]', '', lowercase_string)
    
  return processed_string

def get_dictionnary(df):
  dictionnary=[]
  for i in df[0]:
    for j in i.split(" "):
      dictionnary.append(j)

  dictionnary=set(dictionnary)
  return(dictionnary)



def ruin_grammar(sentence, dictionnary):
  words=sentence.split(" ")
  #remove
  if len(words)>4:
    idx0 = random.sample(range(0, len(words)), 1)[0]
    words.pop(idx0)
  length=len(words)
  #reorder
  idx1,idx2 = random.sample(range(0, length), 2)
  words[idx1], words[idx2] = words[idx2], words[idx1]
  
  #insert
  idx3 = random.sample(range(0, len(dictionnary)), 1)[0]
  idx4 = random.sample(range(0, length), 2)[0]
  words.insert(idx4,list(dictionnary)[idx3])
  out="".join([i+" " for i in words])
  return(out[0:-1])
