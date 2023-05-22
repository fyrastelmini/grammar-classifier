import re
import random
from tensorflow.keras.callbacks import Callback
import numpy as np

def preprocess(sentence):

  lowercase_string = sentence.lower()

  processed_string = re.sub(r'[.,?!$%^:;]', '', lowercase_string)
    
  return processed_string
  
  
  
def preprocess_nolower(sentence):

  processed_string = re.sub(r'[.,?!$%^:;]', '', sentence)
    
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
  

class TestAccuracyCallback(Callback):
    def __init__(self, test_data,model):
        super(TestAccuracyCallback, self).__init__()
        self.test_data = test_data
        self.model = model
    
    def on_epoch_end(self, epoch, logs=None):
        X_test, y_test = self.test_data
        y_test = y_test.reshape((y_test.shape[0],1))
        test_predictions = np.round(self.model.predict(X_test))
        test_accuracy = (y_test == test_predictions).mean()
        print(f'Test Accuracy: {test_accuracy:.4f}')
