import re
import random
from tensorflow.keras.callbacks import Callback
import numpy as np

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
  

class ValidationAccuracyCallback(Callback):
    def __init__(self, validation_data,model):
        super(ValidationAccuracyCallback, self).__init__()
        self.validation_data = validation_data
        self.model = model
    
    def on_epoch_end(self, epoch, logs=None):
        X_test, y_test = self.validation_data
        y_test = y_test.reshape((y_test.shape[0],1))
        val_predictions = np.round(self.model.predict(X_test))
        val_accuracy = (y_test == val_predictions).mean()
        print(y_test.shape,val_predictions.shape)
        print(f'Validation Accuracy: {val_accuracy:.4f}')
