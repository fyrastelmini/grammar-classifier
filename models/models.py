import tensorflow as tf
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier


class Classifier(tf.keras.Model):
    def __init__(self, input_sz, embedding_dim):
        super(Classifier, self).__init__()
        self.embedding = tf.keras.layers.Embedding(input_sz, embedding_dim)
        self.conv1d = tf.keras.layers.Conv1D(256, 8, activation='relu')
        self.layer_norm = tf.keras.layers.LayerNormalization()
        self.dropout = tf.keras.layers.Dropout(0.5)
        self.lstm = tf.keras.layers.LSTM(28)
        self.dense = tf.keras.layers.Dense(1, activation='sigmoid')

    def call(self, x):
        x = self.embedding(x)
        x = self.conv1d(x)
        x = self.dropout(x)
        x = self.lstm(x)
        output = self.dense(x)
        return output
        
def get_ml_model(name):
	  if name == "RF:
	    return RandomForestClassifier(max_depth=16, random_state=0)
	  elif name == "NB":
	    return GaussianNB()
	  elif name == "LR":
	    return LogisticRegression()
	  elif name == "SVC":
	    return SVC()
	  elif name == "GB":
	    return GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=24, random_state=0)
	  elif name == "MLP":
	    return MLPClassifier()
	  else:
	    return 'invalid name: '+name

